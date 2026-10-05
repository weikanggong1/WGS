"""Write native R objects without starting an R process.

SPDX-License-Identifier: GPL-3.0-only
Uses the MIT-licensed rdata conversion and XDR APIs:
https://github.com/vnmabus/rdata. The S4 writer follows R's serialization
representation (an S4 marker followed by its slot attributes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class RAttributed:
    value: Any
    attributes: Mapping[str, Any]
    object_flag: bool = False


@dataclass
class RMatrix:
    values: Any
    columns: Sequence[str] | None = None
    row_names: Sequence[str] | None = None
    mode: str = "double"


@dataclass
class RDataFrame:
    columns: Mapping[str, Any]
    row_names: Sequence[int | str]


@dataclass
class RFactor:
    values: Sequence[str]
    levels: Sequence[str]


@dataclass
class RS4:
    class_name: str
    slots: Mapping[str, Any]
    package: str = "Matrix"


@dataclass
class RSymbol:
    name: str


@dataclass
class RCall:
    name: str
    arguments: Sequence[tuple[str | None, Any]] = field(default_factory=tuple)


def _constructors():
    from rdata.conversion.to_r import DEFAULT_CLASS_MAP, build_r_object, build_r_list
    from rdata.parser import RObjectType

    def attributed(data, converter):
        attributes = dict(data.attributes)
        if isinstance(data.value, Mapping):
            if not all(isinstance(key, str) for key in data.value):
                raise ValueError("named R list keys must be strings")
            # Register each attribute once. Converting an already named dict
            # and then replacing its names can leave a REF to a discarded tag.
            attrs = {"names": np.asarray(list(data.value), dtype=str)}
            attrs.update(attributes)
            attributes = attrs
            obj = converter.convert_to_r_object(list(data.value.values()))
        else:
            obj = converter.convert_to_r_object(data.value)
        pairs = []
        original = obj.attributes
        while original is not None and original.info.type == RObjectType.LIST:
            tag = original.tag
            resolved = tag.referenced_object if tag.info.type == RObjectType.REF else tag
            name = resolved.value.value.decode(converter.encoding)
            if name not in attributes:
                pairs.append((tag, original.value[0]))
            original = original.value[1]
        pairs.extend((converter.convert_to_r_sym(name), converter.convert_to_r_object(value))
                     for name, value in attributes.items())
        attrs = build_r_list(pairs) if pairs else None
        return build_r_object(obj.info.type, value=obj.value,
            is_object=data.object_flag or obj.info.object, attributes=attrs, gp=obj.info.gp)

    def matrix(data, converter):
        values = np.asarray(data.values, dtype=object if data.mode == "list" else None)
        if values.ndim != 2:
            raise ValueError("RMatrix values must be two-dimensional")
        nr, nc = values.shape
        if data.columns is not None and len(data.columns) != nc:
            raise ValueError("matrix column names do not match its shape")
        if data.row_names is not None and len(data.row_names) != nr:
            raise ValueError("matrix row names do not match its shape")
        if data.mode == "list":
            value = [converter.convert_to_r_object(x) for x in values.ravel(order="F")]
            obj = build_r_object(RObjectType.VEC, value=value)
        elif data.mode == "double":
            obj = converter.convert_to_r_object(np.asarray(values, dtype=np.float64).ravel(order="F"))
        elif data.mode == "integer":
            obj = converter.convert_to_r_object(np.asarray(values, dtype=np.int32).ravel(order="F"))
        elif data.mode == "character":
            obj = converter.convert_to_r_object(np.asarray(values, dtype=str).ravel(order="F"))
        else:
            raise ValueError("unsupported RMatrix mode")
        attrs = {"dim": np.asarray([nr, nc], dtype=np.int32)}
        if data.columns is not None or data.row_names is not None:
            attrs["dimnames"] = [None if data.row_names is None else np.asarray(data.row_names, dtype=str),
                                  None if data.columns is None else np.asarray(data.columns, dtype=str)]
        return build_r_object(obj.info.type, value=obj.value,
                              attributes=converter.convert_to_r_attributes(attrs))

    def dataframe(data, converter):
        lengths = [len(column.values) if isinstance(column, RFactor) else len(column) for column in data.columns.values()]
        if any(length != len(data.row_names) for length in lengths):
            raise ValueError("data.frame columns and row.names must be aligned")
        attrs = {"names": np.asarray(list(data.columns), dtype=str),
                 "row.names": np.asarray(data.row_names, dtype=np.int32 if all(isinstance(x, (int, np.integer)) for x in data.row_names) else str),
                 "class": "data.frame"}
        return build_r_object(RObjectType.VEC,
            value=[converter.convert_to_r_object(column) for column in data.columns.values()],
            is_object=True, attributes=converter.convert_to_r_attributes(attrs))

    def factor(data, converter):
        levels = list(data.levels)
        lookup = {value: index + 1 for index, value in enumerate(levels)}
        if len(lookup) != len(levels):
            raise ValueError("factor levels must be distinct")
        try:
            codes = np.asarray([lookup[value] for value in data.values], dtype=np.int32)
        except KeyError:
            raise ValueError("factor value is absent from its levels") from None
        return build_r_object(RObjectType.INT, value=codes, is_object=True,
            attributes=converter.convert_to_r_attributes({"levels": np.asarray(levels, dtype=str), "class": "factor"}))

    def s4(data, converter):
        attrs = dict(data.slots)
        attrs["class"] = RAttributed(np.asarray([data.class_name], dtype=str), {"package": data.package})
        return build_r_object(RObjectType.S4, is_object=True, gp=16,
                              attributes=converter.convert_to_r_attributes(attrs))

    def symbol(data, converter):
        return converter.convert_to_r_sym(data.name)

    def call(data, converter):
        # LANG writes its function symbol before the argument pairlist. Register
        # that symbol first so repeated argument/tag symbols use correct refs.
        function = converter.convert_to_r_sym(data.name)
        if data.arguments:
            tail = build_r_list([
                converter.convert_to_r_object(value) if name is None else
                (converter.convert_to_r_sym(name), converter.convert_to_r_object(value))
                for name, value in data.arguments
            ])
        else:
            tail = build_r_object(RObjectType.NILVALUE)
        return build_r_object(RObjectType.LANG, value=(function, tail))

    return dict(DEFAULT_CLASS_MAP) | {RAttributed: attributed, RMatrix: matrix,
        RDataFrame: dataframe, RFactor: factor, RS4: s4, RSymbol: symbol, RCall: call}


def write_r_object(path, value: Any, *, object_name: str | None = None,
                   compression: bool = True) -> None:
    """Write gzip XDR .Rdata/.rda or .rds, preserving explicit R attributes.

    Rdata stores one named object and requires ``object_name``. RDS stores the
    object directly. None is R NULL; an empty Python list is an empty R list.
    S4 Matrix slots are serialized directly and retain sparse storage.
    """
    from rdata.conversion import convert_python_to_r_data
    from rdata.parser import RObjectType
    from rdata.unparser._xdr import UnparserXDR
    from rdata.unparser._unparser import pack_r_object_info

    class S4Unparser(UnparserXDR):
        def unparse_r_object(self, obj):
            if obj.info.type != RObjectType.S4:
                return super().unparse_r_object(obj)
            self.unparse_int(pack_r_object_info(obj.info))
            if obj.info.attributes:
                self.unparse_r_object(obj.attributes)

    path = Path(path)
    extension = path.suffix.lower()
    if extension not in (".rds", ".rdata", ".rda"):
        raise ValueError("native R output filename must end in .Rdata, .rda or .rds")
    rda = extension != ".rds"
    if rda and (not isinstance(object_name, str) or not object_name):
        raise ValueError("Rdata output requires a nonempty object_name")
    payload = {object_name: value} if rda else value
    parsed = convert_python_to_r_data(payload, constructor_dict=_constructors(),
        file_type="rda" if rda else "rds", format_version=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if compression else open
    # Readers can keep using a completed file while a long null-model object
    # is serialized; publish the replacement only after the stream is closed.
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with opener(temporary_path, "wb") as stream:
            if rda:
                stream.write(b"RDX3\n")
            S4Unparser(stream).unparse_r_data(parsed)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def sparse_matrix(matrix, *, class_name="dgCMatrix", uplo="U", dimnames=None) -> RS4:
    """Represent a SciPy sparse matrix as Matrix's compressed-column S4 class."""
    from scipy import sparse
    if class_name not in ("dgCMatrix", "dsCMatrix"):
        raise ValueError("sparse class must be dgCMatrix or dsCMatrix")
    csc = sparse.csc_matrix(matrix)
    if class_name == "dsCMatrix":
        if csc.shape[0] != csc.shape[1] or uplo not in ("U", "L"):
            raise ValueError("symmetric sparse Matrix requires a square shape and U/L uplo")
        csc = sparse.triu(csc, format="csc") if uplo == "U" else sparse.tril(csc, format="csc")
    csc.sum_duplicates();csc.sort_indices()
    slots = {"i": np.asarray(csc.indices, dtype=np.int32),
             "p": np.asarray(csc.indptr, dtype=np.int32),
             "Dim": np.asarray(csc.shape, dtype=np.int32),
             "Dimnames": [None, None] if dimnames is None else
                         [None if names is None else np.asarray(names, dtype=str) for names in dimnames],
             "x": np.asarray(csc.data, dtype=np.float64)}
    if class_name == "dsCMatrix":
        slots["uplo"] = uplo
    slots["factors"] = []
    return RS4(class_name, slots)


def dense_s4_matrix(matrix, *, dimnames=None) -> RS4:
    """Represent a dense matrix as Matrix's dgeMatrix without expanding a GRM."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("dense_s4_matrix requires a matrix")
    return RS4("dgeMatrix", {"x": value.ravel(order="F"),
        "Dim": np.asarray(value.shape, dtype=np.int32),
        "Dimnames": [None, None] if dimnames is None else
                    [None if names is None else np.asarray(names, dtype=str) for names in dimnames],
        "factors": []})


def association_object(result, *, kind: str, layout="phewas"):
    """Convert records into the reference PheWAS or base object topology.

    The computation API always groups rows by trait. Base STAARpipeline has
    one trait and returns each matrix/data.frame directly; PheWAS retains the
    trait list. Empty masks are R NULL in either layout.
    """
    if layout not in ("phewas", "base"):
        raise ValueError("supported association layouts are phewas and base")
    if kind == "individual":
        kind = "singlevariant"
    if kind not in ("coding", "noncoding", "ncrna", "singlevariant", "sliding"):
        raise ValueError("unsupported association kind")

    def matrix_rows(rows, *, row_label="results_temp"):
        if not rows:
            return None
        columns = list(rows[0])
        if any(list(row) != columns for row in rows):
            raise ValueError("all result rows must have identical ordered columns")
        values = np.empty((len(rows), len(columns)), dtype=object)
        for i, row in enumerate(rows):
            for j, name in enumerate(columns):
                value = row[name]
                if (kind == "ncrna" or kind == "noncoding" and row.get("Category") in
                    ("upstream", "downstream", "UTR", "ncRNA")) and name in ("Chr", "#SNV"):
                    # Upstream first builds an atomic four-cell vector. Gene
                    # and category strings coerce its chromosome/count cells
                    # to character before numeric statistic lists are joined.
                    value = str(value) if isinstance(value, str) else format(float(value), ".15g")
                elif name == "#SNV":
                    value = int(value)
                elif layout == "base" and name == "Chr":
                    # Tutorial array-to-chromosome mapping uses which.max,
                    # whose scalar is integer; PheWAS reference calls use a
                    # numeric chromosome argument. Preserve that distinction.
                    value = int(value)
                elif name not in ("Gene name", "Category"):
                    value = float(value)
                values[i, j] = value
        row_names = getattr(rows, "matrix_row_names",
                            None if row_label is None else [row_label] * len(rows))
        return RMatrix(values, columns, row_names, mode="list")

    def dataframe_rows(rows):
        if not rows:
            return None
        row_names = getattr(rows, "row_names", None)
        levels = getattr(rows, "factor_levels", None)
        if row_names is None or not levels or any(key not in levels for key in ("REF", "ALT")):
            raise ValueError("exact individual output requires TraitRows row_names and REF/ALT factor_levels")
        columns = {}
        for name in rows[0]:
            values = [row[name] for row in rows]
            if name in ("REF", "ALT"):
                columns[name] = RFactor(values, levels[name])
            else:
                columns[name] = np.asarray(values, dtype=np.int32 if name == "N" else np.float64)
        return RDataFrame(columns, row_names)

    def trait_output(values):
        if layout == "base":
            if len(values) != 1:
                raise ValueError("base association layout requires exactly one trait")
            return values[0]
        return values

    if kind == "singlevariant":
        return trait_output([dataframe_rows(rows) for rows in result])
    if isinstance(result, Mapping):
        output = {}
        for category, traits in result.items():
            category_rows = []
            for index, rows in enumerate(traits):
                has_disruptive = bool(result.get("disruptive_missense", [[]] * len(traits))[index])
                label = "results_m" if category == "missense" and has_disruptive else "results_temp"
                if category == "disruptive_missense" and bool(result.get("missense", [[]] * len(traits))[index]):
                    label = None
                category_rows.append(matrix_rows(rows, row_label=label))
            output[category] = trait_output(category_rows)
        return output
    return trait_output([matrix_rows(rows) for rows in result])


def write_association_output(path, result, *, kind: str, object_name=None,
                             layout="phewas") -> None:
    """Write original-style association output and tutorial object names."""
    names = {"coding": "results_coding", "noncoding": "results_noncoding",
             "ncrna": "results_noncoding", "sliding": "results_sliding_window",
             "singlevariant": "results_individual_analysis", "individual": "results_individual_analysis"}
    value = association_object(result, kind=kind, layout=layout)
    default_name = "results_ncRNA" if kind == "ncrna" and layout == "base" else names[kind]
    write_r_object(path, value, object_name=object_name or default_name)


def association_batch_object(results, *, kind: str, layout="phewas"):
    """Follow tutorial append/rbind operations for ordered genomic jobs.

    Named list append retains duplicate category names. Sliding matrices are
    row-bound separately for each trait. A single job is unchanged. Multiple
    individual jobs require the upstream data.frame grouping metadata and are
    deliberately rejected rather than losing factor/row-name information.
    """
    results = list(results)
    if not results:
        raise ValueError("association batch requires at least one job")
    parts = [association_object(result, kind=kind, layout=layout) for result in results]
    if len(parts) == 1:
        return parts[0]
    if kind in ("singlevariant", "individual"):
        raise ValueError("multiple individual jobs cannot share one output file")
    if layout == "base" and kind in ("sliding", "ncrna"):
        matrices = [part for part in parts if part is not None]
        if not matrices:
            return None
        columns = matrices[0].columns
        if any(matrix.columns != columns or matrix.mode != "list" for matrix in matrices):
            raise ValueError("base matrices must have identical ordered columns")
        values = np.concatenate([matrix.values for matrix in matrices], axis=0)
        row_names = None
        if any(matrix.row_names is not None for matrix in matrices):
            row_names = [name for matrix in matrices for name in
                         (matrix.row_names if matrix.row_names is not None else [""] * len(matrix.values))]
        return RMatrix(values, columns, row_names, mode="list")
    if kind == "sliding":
        n_traits = len(parts[0])
        if any(len(part) != n_traits for part in parts):
            raise ValueError("sliding jobs must use the same ordered trait list")
        merged = []
        for trait in range(n_traits):
            matrices = [part[trait] for part in parts if part[trait] is not None]
            if not matrices:
                merged.append(None)
                continue
            columns = matrices[0].columns
            if any(matrix.columns != columns or matrix.mode != "list" for matrix in matrices):
                raise ValueError("sliding matrices must have identical ordered columns")
            values = np.concatenate([matrix.values for matrix in matrices], axis=0)
            row_names = None
            if any(matrix.row_names is not None for matrix in matrices):
                row_names = [name for matrix in matrices for name in
                    (matrix.row_names if matrix.row_names is not None else [""] * len(matrix.values))]
            merged.append(RMatrix(values, columns, row_names, mode="list"))
        return merged
    values, names = [], []
    named = False
    for part in parts:
        if isinstance(part, Mapping):
            named = True
            values.extend(part.values())
            names.extend(part.keys())
        else:
            if layout == "base":
                # R append(NULL, selected_matrix) appends its cells, stripping
                # dimensions. Tutorials call all_categories; preserve the
                # selected-category append semantics when explicitly used.
                cells = [] if part is None else list(np.asarray(part.values).ravel(order="F"))
                values.extend(cells)
                names.extend([""] * len(cells))
            else:
                values.extend(part)
                names.extend([""] * len(part))
    return RAttributed(values, {"names": np.asarray(names, dtype=str)}) if named else values


def write_association_batch(path, results, *, kind: str, object_name=None,
                            layout="phewas") -> None:
    """Write ordered jobs to one original-style file, retaining repeated names."""
    names = {"coding": "results_coding", "noncoding": "results_noncoding",
             "ncrna": "results_noncoding", "sliding": "results_sliding_window",
             "singlevariant": "results_individual_analysis", "individual": "results_individual_analysis"}
    value = association_batch_object(results, kind=kind, layout=layout)
    default_name = "results_ncRNA" if kind == "ncrna" and layout == "base" else names[kind]
    write_r_object(path, value, object_name=object_name or default_name)
