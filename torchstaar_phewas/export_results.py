"""Export native independent association files to private per-trait CSVs.

SPDX-License-Identifier: GPL-3.0-only
This is output postprocessing: it never recomputes association statistics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import unicodedata

import numpy as np
from rdata.parser import RObjectType, parse_file
from rdata.conversion._conversion import convert_char
from rdata.missing import is_na


_KINDS = frozenset(("individual", "coding", "noncoding", "ncrna"))
_SINGLE_COLUMNS = ("CHR", "POS", "REF", "ALT", "ALT_AF", "MAF", "N", "pvalue",
                   "pvalue_log10", "Score", "Score_se", "Est", "Est_se")
_GENE_COLUMNS = ("Gene name", "Chr", "Category", "#SNV", "cMAC") + tuple(
    field for method, combined in (("SKAT", "STAAR-S"), ("Burden", "STAAR-B"), ("ACAT-V", "STAAR-A"))
    for beta in ("1,25", "1,1") for field in (f"{method}({beta})", f"{combined}({beta})")) + ("ACAT-O", "STAAR-O")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_reference(node):
    visited = set()
    while node is not None and node.info.type == RObjectType.REF:
        if id(node) in visited or node.referenced_object is None:
            raise ValueError("invalid native R reference")
        visited.add(id(node))
        node = node.referenced_object
    return node


@dataclass
class _Table:
    columns: tuple[str, ...]
    rows: int
    cell: object


class _NativeReader:
    """Visit parser objects directly; named list slots never become a dict."""
    def __init__(self, parsed):
        self.parsed = parsed
        self.encoding = parsed.extra.encoding or "utf-8"
        if isinstance(self.encoding, bytes):
            self.encoding = self.encoding.decode("ascii")
        self.tables, self.slots = [], []

    def text(self, node):
        node = _resolve_reference(node)
        if node.info.type == RObjectType.SYM:
            return self.text(node.value)
        if node.info.type != RObjectType.CHAR:
            raise ValueError("native R name must be a character scalar")
        result = convert_char(node, default_encoding=self.encoding)
        if isinstance(result, bytes):
            result = result.decode("utf-8")
        return result

    def attributes(self, node):
        result = {}
        current = _resolve_reference(node.attributes)
        while current is not None and current.info.type != RObjectType.NILVALUE:
            if current.info.type != RObjectType.LIST or current.tag is None:
                raise ValueError("invalid native R attribute list")
            name = self.text(current.tag)
            if name in result:
                raise ValueError("duplicate native R attribute")
            result[name] = _resolve_reference(current.value[0])
            current = _resolve_reference(current.value[1])
        return result

    def strings(self, node):
        node = _resolve_reference(node)
        if node is None or node.info.type == RObjectType.NILVALUE:
            return None
        if node.info.type != RObjectType.STR:
            raise ValueError("native R labels must be a character vector")
        return tuple(self.text(item) for item in node.value)

    def vector(self, node):
        node = _resolve_reference(node)
        attrs = self.attributes(node)
        classes = self.strings(attrs.get("class")) or ()
        if "factor" in classes:
            levels = self.strings(attrs.get("levels"))
            if levels is None:
                raise ValueError("native R factor has no levels")
            codes = np.asanyarray(node.value)
            values = []
            for code in codes:
                if np.ma.is_masked(code):
                    values.append(None)
                elif not 1 <= int(code) <= len(levels):
                    raise ValueError("native R factor code is outside its levels")
                else:
                    values.append(levels[int(code) - 1])
            return values
        if node.info.type == RObjectType.STR:
            return [self.text(item) for item in node.value]
        if node.info.type in (RObjectType.REAL, RObjectType.INT, RObjectType.LGL):
            values = np.asanyarray(node.value)
            if node.info.type == RObjectType.REAL:
                # R NA_real_ has its own NaN payload. It is a missing cell;
                # ordinary NaN remains a visible NaN in the CSV.
                missing = is_na(values)
                if np.any(missing):
                    return np.ma.array(values, mask=missing, copy=False)
            return values
        if node.info.type == RObjectType.VEC:
            return [self.scalar(item) for item in node.value]
        raise ValueError("unsupported native association vector")

    def scalar(self, node):
        node = _resolve_reference(node)
        if node is None or node.info.type == RObjectType.NILVALUE:
            return None
        values = self.vector(node)
        if len(values) != 1:
            raise ValueError("native association matrix cell is not scalar")
        return values[0]

    def select(self, extension, kind, object_name):
        root = _resolve_reference(self.parsed.object)
        if extension == ".rds":
            return root
        objects = []
        while root is not None and root.info.type != RObjectType.NILVALUE:
            if root.info.type != RObjectType.LIST or root.tag is None:
                raise ValueError("native workspace must contain named objects")
            objects.append((self.text(root.tag), _resolve_reference(root.value[0])))
            root = _resolve_reference(root.value[1])
        if object_name is None:
            if len(objects) != 1:
                raise ValueError("object_name is required for a workspace with multiple objects")
            return objects[0][1]
        selected = [node for name, node in objects if name == object_name]
        if len(selected) != 1:
            raise ValueError("object_name must select exactly one native workspace object")
        return selected[0]

    def visit(self, node, path=(), names=(), active=None):
        node = _resolve_reference(node)
        if node is None or node.info.type == RObjectType.NILVALUE:
            self.slots.append(dict(ordinal_path=list(path), names=list(names), null=True, rows=0))
            return
        active = set() if active is None else active
        if len(path) > 64 or id(node) in active:
            raise ValueError("cyclic or excessively nested native association list")
        attrs = self.attributes(node)
        classes = self.strings(attrs.get("class")) or ()
        table = None
        if "data.frame" in classes:
            columns = self.strings(attrs.get("names"))
            if node.info.type != RObjectType.VEC or columns is None or len(columns) != len(node.value):
                raise ValueError("native data.frame columns are not aligned")
            vectors = [self.vector(item) for item in node.value]
            lengths = {len(column) for column in vectors}
            if len(lengths) > 1:
                raise ValueError("native data.frame has unequal column lengths")
            rows = next(iter(lengths), 0)
            table = _Table(columns, rows, lambda row, column: vectors[column][row])
        elif "dim" in attrs:
            shape = self.vector(attrs["dim"])
            if len(shape) != 2 or any(np.ma.is_masked(x) or int(x) < 0 for x in shape):
                raise ValueError("native association array must be a two-dimensional matrix")
            nr, nc = map(int, shape)
            dimnames = attrs.get("dimnames")
            if dimnames is None or dimnames.info.type != RObjectType.VEC or len(dimnames.value) != 2:
                raise ValueError("native association matrix requires column dimnames")
            columns = self.strings(dimnames.value[1])
            if columns is None or len(columns) != nc:
                raise ValueError("native matrix columns differ from dimnames")
            if node.info.type == RObjectType.VEC:
                values = node.value
                cell = lambda row, column: self.scalar(values[row + column * nr])
            else:
                values = self.vector(node)
                cell = lambda row, column: values[row + column * nr]
            if len(values) != nr * nc:
                raise ValueError("native matrix dimensions differ from its cells")
            table = _Table(columns, nr, cell)
        elif node.info.type == RObjectType.VEC:
            labels = self.strings(attrs.get("names"))
            if labels is not None and len(labels) != len(node.value):
                raise ValueError("native list names differ from its slots")
            active.add(id(node))
            try:
                for index, child in enumerate(node.value):
                    label = None if labels is None else labels[index]
                    self.visit(child, path + (index + 1,), names + (label,), active)
            finally:
                active.remove(id(node))
            if not node.value:
                self.slots.append(dict(ordinal_path=list(path), names=list(names), null=False, rows=0, empty_list=True))
            return
        elif "names" in attrs:
            columns = self.strings(attrs["names"])
            values = self.vector(node)
            if columns is None or len(columns) != len(values):
                raise ValueError("named native statistics vector is not aligned")
            table = _Table(columns, 1, lambda row, column: values[column])
        else:
            raise ValueError("unsupported native association result topology")
        if any(not isinstance(name, str) for name in table.columns):
            raise ValueError("native association columns require nonmissing names")
        self.tables.append(table)
        self.slots.append(dict(ordinal_path=list(path), names=list(names), null=False, rows=table.rows))


def _column_keys(columns):
    seen = Counter()
    keys = []
    for column in columns:
        keys.append((column, seen[column]))
        seen[column] += 1
    return tuple(keys)


def _column_names(values, parameter):
    if not isinstance(values, (list, tuple)) or any(not isinstance(value, str) for value in values):
        raise TypeError(parameter + " must be a sequence of column names")
    return tuple(values)


def _cell_text(value):
    if value is None or np.ma.is_masked(value):
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return format(value, ".17g")
    if isinstance(value, str):
        return value
    raise ValueError("association cells must be scalar strings, numbers, logicals or missing values")


def _is_probability_column(name):
    return (name in ("pvalue", "ACAT-O", "STAAR-O", "STAAR-B")
            or name.startswith(("SKAT(", "Burden(", "ACAT-V(", "STAAR-S(", "STAAR-B(", "STAAR-A(")))


def _csv_records(reader, table_keys, union, positions, excluded):
    for table, keys in zip(reader.tables, table_keys):
        for row in range(table.rows):
            record, originals = [""] * len(union), [None] * len(union)
            for column, key in enumerate(keys):
                if key[0] not in excluded:
                    value = table.cell(row, column)
                    index = positions[key]
                    record[index], originals[index] = _cell_text(value), value
            yield record, originals


def _verify_csv(target, reader, table_keys, union, positions, excluded):
    counts = dict(checked_cells=0, numeric_cells=0, p_cells=0, logp_cells=0,
                  invalid_p_cells=0, unparsed_probability_cells=0, nonfinite_logp_cells=0)
    maximum_logp_error, maximum_stored_logp_error = 0.0, 0.0
    with target.open("r", encoding="utf-8", newline="") as stream:
        actual_rows = csv.reader(stream)
        header = next(actual_rows, None)
        if header != [key[0] for key in union] or any(column in excluded for column in header):
            raise RuntimeError("CSV header failed the original-column verification")
        for expected, originals in _csv_records(reader, table_keys, union, positions, excluded):
            actual = next(actual_rows, None)
            if actual != expected:
                raise RuntimeError("CSV row failed scalar roundtrip verification")
            for column, (cell, original) in enumerate(zip(actual, originals)):
                counts["checked_cells"] += 1
                if isinstance(original, np.generic):
                    original = original.item()
                if isinstance(original, bool) or original is None or np.ma.is_masked(original):
                    continue
                if isinstance(original, int):
                    if int(cell) != original:
                        raise RuntimeError("CSV integer changed in roundtrip")
                    counts["numeric_cells"] += 1
                elif isinstance(original, float):
                    restored = float(cell)
                    same = (math.isnan(original) and math.isnan(restored)) or (
                        restored == original and (original != 0 or
                            math.copysign(1, original) == math.copysign(1, restored)))
                    if not same:
                        raise RuntimeError("CSV floating value changed in roundtrip")
                    counts["numeric_cells"] += 1
                name = union[column][0]
                if _is_probability_column(name) or name == "pvalue_log10":
                    # Character matrices can contain numerical probability
                    # cells. Their original strings remain byte-for-byte;
                    # only designated statistical columns are parsed here.
                    try:
                        source_value, restored = float(original), float(cell)
                    except (ValueError, TypeError, OverflowError):
                        counts["unparsed_probability_cells"] += 1
                        continue
                    if _is_probability_column(name):
                        counts["p_cells"] += 1
                        if not math.isfinite(source_value) or not 0 <= source_value <= 1:
                            counts["invalid_p_cells"] += 1
                        elif source_value > 0:
                            maximum_logp_error = max(maximum_logp_error,
                                abs(math.log10(source_value) - math.log10(restored)))
                    else:
                        counts["logp_cells"] += 1
                        if math.isfinite(source_value):
                            maximum_stored_logp_error = max(maximum_stored_logp_error,
                                abs(source_value - restored))
                        else:
                            counts["nonfinite_logp_cells"] += 1
        if next(actual_rows, None) is not None:
            raise RuntimeError("CSV has unexpected additional rows")
    return dict(roundtrip_passed=True, numeric_roundtrip_passed=True,
                pvalue_log10_present="pvalue_log10" in header, excluded_columns_absent=True,
                maximum_logp_error=maximum_logp_error,
                maximum_stored_logp_error=maximum_stored_logp_error, **counts)


def _write_csv(source, target, kind, object_name, excluded, empty_columns):
    reader = _NativeReader(parse_file(source))
    reader.visit(reader.select(source.suffix.lower(), kind, object_name))
    union, positions, table_keys = [], {}, []
    for table in reader.tables:
        keys = _column_keys(table.columns)
        table_keys.append(keys)
        for key in keys:
            if key[0] not in excluded and key not in positions:
                positions[key] = len(union)
                union.append(key)
    empty_schema = None
    if not union:
        if reader.tables and any(table.columns for table in reader.tables):
            raise ValueError("exclude_columns removes all association columns")
        columns = empty_columns
        if columns is None:
            columns = _SINGLE_COLUMNS if kind == "individual" else _GENE_COLUMNS
            empty_schema = "default_unannotated"
        else:
            empty_schema = "provided"
        union = [key for key in _column_keys(columns) if key[0] not in excluded]
        positions = {key: index for index, key in enumerate(union)}
        if not union:
            raise ValueError("an empty CSV requires at least one retained column")
    for keys in table_keys:
        order = [positions[key] for key in keys if key[0] not in excluded]
        if order != sorted(order):
            raise ValueError("native tables have incompatible original column orders")
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([key[0] for key in union])
        for record, _ in _csv_records(reader, table_keys, union, positions, excluded):
            writer.writerow(record)
        stream.flush()
        os.fsync(stream.fileno())
    verification = _verify_csv(target, reader, table_keys, union, positions, excluded)
    return dict(rows=sum(table.rows for table in reader.tables), columns=[key[0] for key in union],
                slots=reader.slots, null_slots=sum(slot.get("null", False) for slot in reader.slots),
                empty_schema=empty_schema, **verification)


def _name_key(value):
    return unicodedata.normalize("NFC", value).casefold()


def _trait_name(name):
    if (not isinstance(name, str) or not name or name in (".", "..") or not name.strip()
            or any(character in name for character in ("/", "\\", ":", "\0"))
            or any(ord(character) < 32 for character in name)):
        raise ValueError("trait name must be a nonempty safe directory component")
    return name


def _plan(traits, excluded, empty_columns):
    if not isinstance(traits, (list, tuple)) or not traits:
        raise ValueError("traits must be a nonempty ordered manifest")
    names, plan = set(), []
    for trait in traits:
        if not isinstance(trait, dict) or set(trait) - {"name", "native_files", "exclude_columns"}:
            raise ValueError("unknown trait manifest fields")
        name = _trait_name(trait.get("name"))
        key = _name_key(name)
        if key in names:
            raise ValueError("trait output directory collision")
        names.add(key)
        removed = frozenset(excluded + _column_names(trait.get("exclude_columns", ()), "exclude_columns"))
        files = trait.get("native_files")
        if not isinstance(files, (list, tuple)) or not files:
            raise ValueError("every trait requires ordered native_files")
        destinations, entries = set(), []
        for entry in files:
            if not isinstance(entry, dict) or set(entry) - {"path", "kind", "object_name", "empty_columns"}:
                raise ValueError("unknown native file manifest fields")
            if entry.get("kind") not in _KINDS:
                raise ValueError("native kind must be individual, coding, noncoding or ncrna")
            if not isinstance(entry.get("path"), (str, os.PathLike)):
                raise ValueError("native file path is required")
            source = Path(entry["path"]).resolve(strict=True)
            if not source.is_file() or source.suffix.lower() not in (".rdata", ".rda", ".rds"):
                raise ValueError("native input must be an existing Rdata, rda or RDS file")
            object_name = entry.get("object_name")
            if object_name is not None and (not isinstance(object_name, str) or not object_name):
                raise ValueError("object_name must be a nonempty string or omitted")
            columns = entry.get("empty_columns", empty_columns)
            if columns is not None:
                columns = _column_names(columns, "empty_columns")
            csv_name = source.stem + ".csv"
            for filename in (source.name, csv_name):
                output_key = _name_key(filename)
                if output_key in destinations:
                    raise ValueError("native basename or CSV stem collision")
                destinations.add(output_key)
            entries.append(dict(source=source, kind=entry["kind"], object_name=object_name,
                                csv_name=csv_name, excluded=removed, empty_columns=columns))
        plan.append((name, entries))
    return plan


@contextmanager
def _export_lock(root):
    path = root / ".torchstaar-export.lock"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise RuntimeError("results directory already has an export in progress or a stale export lock") from None
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        path.unlink(missing_ok=True)


def _existing_state(target, expected_sha):
    if target.is_symlink():
        raise ValueError("export destination cannot be a symbolic link")
    if target.exists():
        if not target.is_file() or _sha256(target) != expected_sha:
            raise FileExistsError("existing export differs; no result was overwritten")
        return "reused"
    return "created"


def _publish(staged, target, expected_sha):
    state = _existing_state(target, expected_sha)
    if state == "created":
        # A hard link publishes the completed temporary file atomically and
        # refuses a concurrent destination; unlike replace, it cannot overwrite.
        try:
            os.link(staged, target)
        except FileExistsError:
            state = _existing_state(target, expected_sha)
    return state


def export_results(traits, *, results_directory, exclude_columns=(), empty_columns=None):
    """Keep native files and export one same-stem CSV per manifest file.

    Each ordered trait dictionary contains ``name`` and ``native_files``;
    each file contains ``path``, ``kind`` and optional ``object_name`` or
    ``empty_columns``. Exclusions may be supplied globally or per trait.
    Trait names are private directory labels. CSV rows follow original list
    slot/table/row order; duplicate named slots are retained. Float values use
    17 significant digits and native log-P columns are copied unchanged.

    A matching SHA reuses an existing file; differing existing files are
    rejected. Each file is published atomically. A failed multi-file export
    may leave completed files, which an identical retry verifies and reuses.
    Returned file/slot metadata is private and is not a public benchmark.
    """
    excluded = _column_names(exclude_columns, "exclude_columns")
    if empty_columns is not None:
        empty_columns = _column_names(empty_columns, "empty_columns")
    plan = _plan(traits, excluded, empty_columns)
    root = Path(results_directory).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    reports = []
    with _export_lock(root):
        for name, entries in plan:
            destination = root / name
            if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
                raise ValueError("trait destination must be an ordinary directory")
            with tempfile.TemporaryDirectory(prefix=".export-stage-", dir=root) as temporary:
                stage = Path(temporary)
                files, publications = [], []
                for entry in entries:
                    source = entry["source"]
                    native_sha = _sha256(source)
                    csv_path = stage / entry["csv_name"]
                    details = _write_csv(source, csv_path, entry["kind"], entry["object_name"],
                                         entry["excluded"], entry["empty_columns"])
                    native_path = stage / source.name
                    shutil.copyfile(source, native_path)
                    with native_path.open("rb") as stream:
                        os.fsync(stream.fileno())
                    if _sha256(source) != native_sha or _sha256(native_path) != native_sha:
                        raise RuntimeError("native source changed during export")
                    csv_sha = _sha256(csv_path)
                    publications.extend(((native_path, destination / source.name, native_sha),
                                         (csv_path, destination / entry["csv_name"], csv_sha)))
                    files.append(dict(kind=entry["kind"], source=str(source),
                        native_file=str(destination / source.name), csv_file=str(destination / entry["csv_name"]),
                        source_sha256=native_sha, native_sha256=native_sha, csv_sha256=csv_sha, **details))
                # Check the entire trait before publishing its first new file.
                for _, target, digest in publications:
                    _existing_state(target, digest)
                destination.mkdir(exist_ok=True, mode=0o700)
                for index, (staged, target, digest) in enumerate(publications):
                    files[index // 2]["native_status" if index % 2 == 0 else "csv_status"] = _publish(staged, target, digest)
                reports.append(dict(name=name, directory=str(destination), native_files=len(files),
                                    csv_files=len(files), rows=sum(item["rows"] for item in files), files=files))
    states = [item[key] for report in reports for item in report["files"] for key in ("native_status", "csv_status")]
    return dict(schema_version=1, completed=True, association_computed=False, traits=reports,
                native_files=sum(report["native_files"] for report in reports),
                csv_files=sum(report["csv_files"] for report in reports),
                created_files=states.count("created"), reused_files=states.count("reused"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export original association files to private per-trait CSVs")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--results-directory", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--exclude-column", action="append", default=[])
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.report:
        reserved = {args.manifest.resolve(), args.results_directory.resolve() / ".torchstaar-export.lock"}
        for name, entries in _plan(manifest, tuple(args.exclude_column), None):
            for entry in entries:
                reserved.update((entry["source"], args.results_directory.resolve() / name / entry["source"].name,
                                 args.results_directory.resolve() / name / entry["csv_name"]))
        if args.report.is_symlink() or args.report.resolve() in reserved:
            raise ValueError("report path collides with an input or association output")
    report = export_results(manifest,
                            results_directory=args.results_directory, exclude_columns=args.exclude_column)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.report.parent,
                                         prefix=".export-report-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        try:
            if args.report.exists():
                if args.report.read_bytes() != temporary.read_bytes():
                    raise FileExistsError("existing export report differs")
            else:
                os.link(temporary, args.report)
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps({key: report[key] for key in ("completed", "native_files", "csv_files", "created_files", "reused_files")}))
    return report


if __name__ == "__main__":
    main()
