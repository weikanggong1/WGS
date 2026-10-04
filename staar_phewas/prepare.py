"""Align private phenotypes, native GDS IDs and an R sparse relationship matrix.

This module prepares inputs only; it performs no association calculation and
never starts R. SPDX-License-Identifier: GPL-3.0-only
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import re
import numpy as np
from scipy import sparse
from .gds import SeqArrayGDS


def _decode_r(node):
    """Decode the data slots needed for native Matrix CSC input."""
    from rdata.parser import RObjectType as T
    while node.info.type == T.REF:
        node = node.referenced_object
    kind = node.info.type
    if kind in (T.NIL, T.NILVALUE):
        return None
    if kind == T.CHAR:
        return None if node.value is None else node.value.decode('utf-8')
    if kind == T.SYM:
        return _decode_r(node.value)
    if kind == T.STR:
        return np.asarray([_decode_r(item) for item in node.value], dtype=str)
    if kind in (T.INT, T.REAL, T.LGL):
        if np.ma.isMaskedArray(node.value) and bool(np.ma.getmaskarray(node.value).any()):
            raise ValueError('relationship matrix slots contain missing values')
        return np.asarray(node.value)
    if kind == T.VEC:
        return [_decode_r(item) for item in node.value]
    if kind in (T.LIST, T.S4):
        current = node.attributes if kind == T.S4 else node
        result = {}
        while current is not None:
            while current.info.type == T.REF:
                current = current.referenced_object
            if current.info.type in (T.NIL, T.NILVALUE):
                break
            if current.info.type != T.LIST or current.tag is None:
                raise ValueError('expected a named R pairlist')
            result[_decode_r(current.tag)] = _decode_r(current.value[0])
            current = current.value[1]
        return result
    raise ValueError('unsupported object in sparse relationship input')


def read_relationship_matrix(path, *, object_name=None):
    """Read a named dsCMatrix/dgCMatrix from Rdata, returning CSC and row IDs."""
    from rdata.parser import parse_file
    objects = _decode_r(parse_file(path).object)
    if not isinstance(objects, dict):
        raise ValueError('relationship input must be a named Rdata workspace')
    if object_name is None:
        candidates = [value for value in objects.values()
                      if isinstance(value, dict) and {'Dim','Dimnames','i','p','x'} <= set(value)]
        if len(candidates) != 1:
            raise ValueError('specify grm_object when the workspace has multiple matrix objects')
        slots = candidates[0]
    else:
        if object_name not in objects:
            raise ValueError('requested relationship object is absent')
        slots = objects[object_name]
    shape = tuple(int(value) for value in slots['Dim'])
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError('relationship matrix must be square')
    names = slots['Dimnames']
    if names is None or names[0] is None or names[1] is None or not np.array_equal(names[0], names[1]):
        raise ValueError('relationship row/column IDs must exist and match')
    matrix = sparse.csc_matrix((np.asarray(slots['x'], dtype=np.float64),
                               np.asarray(slots['i']), np.asarray(slots['p'])), shape=shape)
    matrix.check_format(full_check=True)
    classes = slots.get('class', [])
    if 'dsCMatrix' in classes:
        matrix = matrix + matrix.T - sparse.diags(matrix.diagonal(), format='csc')
    elif 'dgCMatrix' in classes:
        difference = matrix - matrix.T
        if difference.nnz and np.max(np.abs(difference.data)) > 1e-14:
            raise ValueError('relationship matrix is not symmetric')
    else:
        raise ValueError('relationship input must be dsCMatrix or symmetric dgCMatrix')
    if not np.isfinite(matrix.data).all():
        raise ValueError('relationship matrix contains nonfinite values')
    return matrix.tocsc(), np.asarray(names[0], dtype=str)


def _normalize(values, pattern):
    if pattern is None:
        normalized = np.asarray(values, dtype=str)
    else:
        expression = re.compile(pattern)
        if expression.groups != 1:
            raise ValueError('id_pattern must contain exactly one capture group')
        result = []
        for value in values:
            match = expression.search(str(value))
            if match is None:
                raise ValueError('id_pattern does not match every sample ID')
            result.append(match.group(1))
        normalized = np.asarray(result, dtype=str)
    if len(set(normalized)) != len(normalized):
        raise ValueError('sample IDs are duplicated after normalization')
    return normalized


def _numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def prepare_input(*, gds, phenotypes, phenotype_columns, output, id_column='IID',
                  covariate_columns=(), grm=None, grm_object=None, id_pattern=None,
                  exclude=None, exclude_id_index=1, delimiter='whitespace'):
    """Write aligned NPZ in phenotype-table order and return aggregate counts."""
    if not phenotype_columns or len(set(phenotype_columns)) != len(phenotype_columns):
        raise ValueError('phenotype_columns must be nonempty and unique')
    if delimiter not in ('whitespace','tab','comma'):
        raise ValueError('delimiter must be whitespace, tab, or comma')
    with open(phenotypes, newline='') as stream:
        if delimiter == 'whitespace':
            header = next(stream).split()
            table = []
            for line in stream:
                if not line.strip(): continue
                fields = line.split()
                if len(fields) != len(header):
                    raise ValueError('phenotype rows must have the same width as the header')
                table.append(dict(zip(header, fields)))
        else:
            reader = csv.DictReader(stream, delimiter='\t' if delimiter == 'tab' else ',')
            header = reader.fieldnames
            table = list(reader)
    if not set([id_column, *phenotype_columns, *covariate_columns]) <= set(header or []):
        raise ValueError('requested ID/phenotype/covariate columns are missing')
    ids = np.asarray([row[id_column] for row in table], dtype=str)
    if len(set(ids)) != len(ids): raise ValueError('phenotype IDs must be unique')
    y = np.asarray([[_numeric(row[name]) for name in phenotype_columns] for row in table], dtype=np.float64)
    x = np.asarray([[_numeric(row[name]) for name in covariate_columns] for row in table], dtype=np.float64).reshape(len(table),len(covariate_columns))
    complete = np.isfinite(y).all(axis=1) & np.isfinite(x).all(axis=1)
    excluded = set()
    if exclude is not None:
        if exclude_id_index < 0: raise ValueError('exclude_id_index must be nonnegative')
        with open(exclude) as stream:
            for line in stream:
                fields = line.split()
                if not fields: continue
                if len(fields)>1 and exclude_id_index >= len(fields):
                    raise ValueError('exclude_id_index exceeds the exclusion row width')
                value = fields[0] if len(fields)==1 else fields[exclude_id_index]
                if value not in ('FID','IID','#FID','#IID'): excluded.add(value)
    with SeqArrayGDS(gds) as genotype_file:
        canonical_ids = genotype_file.sample_ids()
    gds_ids = _normalize(canonical_ids, id_pattern)
    gds_lookup = {value:index for index,value in enumerate(gds_ids)}
    relationship, grm_lookup = None, None
    if grm is not None:
        relationship, relationship_ids = read_relationship_matrix(grm, object_name=grm_object)
        relationship_ids = _normalize(relationship_ids, id_pattern)
        grm_lookup = {value:index for index,value in enumerate(relationship_ids)}
    keep = np.asarray([complete[index] and value not in excluded and value in gds_lookup
                       and (grm_lookup is None or value in grm_lookup) for index,value in enumerate(ids)])
    aligned_ids = ids[keep]
    if len(aligned_ids) < 2: raise ValueError('fewer than two aligned complete samples')
    gds_rows = np.asarray([gds_lookup[value] for value in aligned_ids], dtype=np.int64)
    values = {'ids':aligned_ids, 'gds_sample_ids':canonical_ids[gds_rows], 'sample_indices':gds_rows,
              'y_raw':y[keep,0] if y.shape[1]==1 else y[keep],
              'phenotype_names':np.asarray(phenotype_columns, dtype=str)}
    if covariate_columns:
        values['covariates'] = np.column_stack((np.ones(len(aligned_ids)), x[keep]))
        values['covariate_names'] = np.asarray(['(Intercept)', *covariate_columns], dtype=str)
    if relationship is not None:
        grm_rows = np.asarray([grm_lookup[value] for value in aligned_ids], dtype=np.int64)
        selected = relationship[grm_rows][:,grm_rows].tocoo()
        diagonal = selected.tocsr().diagonal()
        if (diagonal < 0).any(): raise ValueError('negative GRM diagonal')
        edge = selected.row < selected.col
        values.update(grm_indices=grm_rows, grm_diagonal=diagonal,
                      grm_edge_row=selected.row[edge].astype(np.int64),
                      grm_edge_col=selected.col[edge].astype(np.int64),
                      grm_edge_value=selected.data[edge])
    destination = Path(output); destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **values)
    return {'phenotype_rows':len(ids), 'complete_rows':int(complete.sum()),
            'excluded_ids':len(excluded), 'aligned_samples':len(aligned_ids),
            'number_phenotypes':len(phenotype_columns),
            'grm_edges':len(values.get('grm_edge_value', []))}


def main():
    parser = argparse.ArgumentParser(description='Prepare private aligned STAAR inputs without R')
    for name in ('gds','phenotypes','output'): parser.add_argument('--'+name, required=True)
    parser.add_argument('--phenotype-columns', nargs='+', required=True)
    parser.add_argument('--id-column', default='IID')
    parser.add_argument('--covariate-columns', nargs='*', default=[])
    parser.add_argument('--grm'); parser.add_argument('--grm-object'); parser.add_argument('--id-pattern')
    parser.add_argument('--exclude'); parser.add_argument('--exclude-id-index',type=int,default=1)
    parser.add_argument('--delimiter',choices=['whitespace','tab','comma'],default='whitespace')
    print(json.dumps(prepare_input(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == '__main__': main()
