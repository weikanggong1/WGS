// SPDX-License-Identifier: GPL-3.0-only
// I/O-only adapter for the official CoreArray/PyGDS capsule SDK.
// API provenance: Xiuwen Zheng, CoreArray/pygds, PyGDS.h / PyGDS2.h.
// No GDS implementation, genotype interpretation or statistical code is copied.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <PyGDS2.h>
#include <exception>
#include <cstring>
#include <limits>

static PdAbstractArray bit2_array(PdGDSObj object, C_Int64 *element_count)
{
        char class_name[64] = {};
        GDS_Node_GetClassName(object, class_name, sizeof(class_name)-1);
        if (std::strcmp(class_name, "dBit2") != 0) {
            PyErr_SetString(PyExc_TypeError, "raw Bit2 reads require an unsigned Bit2 array");
            return NULL;
        }
        PdAbstractArray array = object;
        if (GDS_Array_GetBitOf(array) != 2 || !COREARRAY_SV_INTEGER(GDS_Array_GetSVType(array))) {
            PyErr_SetString(PyExc_TypeError, "raw Bit2 reads require a two-bit integer array");
            return NULL;
        }
        C_Int64 total = GDS_Array_GetTotalCount(array);
        int ndim = GDS_Array_DimCnt(array);
        if (ndim < 1 || ndim > GDS_MAX_NUM_DIMENSION) {
            PyErr_SetString(PyExc_RuntimeError, "invalid GDS array dimension count");
            return NULL;
        }
        C_Int32 dimensions[GDS_MAX_NUM_DIMENSION];
        GDS_Array_GetDim(array, dimensions, ndim);
        C_Int64 product = 1;
        for (int i=0; i<ndim; ++i) {
            if (dimensions[i] < 0 || (dimensions[i] &&
                product > std::numeric_limits<C_Int64>::max()/dimensions[i])) {
                PyErr_SetString(PyExc_OverflowError, "invalid or overflowing GDS dimensions");
                return NULL;
            }
            product *= dimensions[i];
        }
        if (total < 0 || product != total) {
            PyErr_SetString(PyExc_RuntimeError, "GDS dimensions differ from container element count");
            return NULL;
        }
        *element_count=total;
        return array;
}

static PyObject *read_array(PdGDSObj object, long long offset, long long count, const char *dtype)
{
    if (offset < 0 || count < 0) {
        PyErr_SetString(PyExc_ValueError, "flat_offset and flat_count must be nonnegative");
        return NULL;
    }
    if (std::strcmp(dtype, "uint8") != 0) {
        PyErr_SetString(PyExc_ValueError, "raw Bit2 reads support output_dtype='uint8'");
        return NULL;
    }
    if (static_cast<unsigned long long>(count) > static_cast<unsigned long long>(PY_SSIZE_T_MAX)) {
        PyErr_SetString(PyExc_OverflowError, "flat_count exceeds Python buffer size");
        return NULL;
    }
    PyObject *result = NULL;
    Py_buffer buffer;
    bool has_buffer = false;
    // Keep the GIL throughout the read so another Python thread cannot close
    // the owning file between official file lookup and the iterator call.
    try {
        C_Int64 total;
        PdAbstractArray array=bit2_array(object,&total);
        if (!array) return NULL;
        if (offset > total || count > total - offset) {
            PyErr_SetString(PyExc_IndexError, "flat read exceeds the array element count");
            return NULL;
        }
        PyObject *numpy = PyImport_ImportModule("numpy");
        if (!numpy) return NULL;
        result = PyObject_CallMethod(numpy, "empty", "Ls", count, "uint8");
        Py_DECREF(numpy);
        if (!result) return NULL;
        if (PyObject_GetBuffer(result, &buffer, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) < 0) {
            Py_DECREF(result);
            return NULL;
        }
        has_buffer = true;
        if (buffer.len != count || buffer.itemsize != 1) {
            PyBuffer_Release(&buffer);
            Py_DECREF(result);
            PyErr_SetString(PyExc_RuntimeError, "NumPy returned an unexpected uint8 buffer");
            return NULL;
        }
        if (count) {
            CdIterator iterator;
            GDS_Iter_Position(array, &iterator, static_cast<C_Int64>(offset));
            GDS_Iter_RData(&iterator, buffer.buf, static_cast<size_t>(count), svUInt8);
        }
        PyBuffer_Release(&buffer);
        return result;
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
    } catch (const char *error) {
        PyErr_SetString(PyExc_RuntimeError, error);
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "CoreArray flat iterator read failed");
    }
    if (has_buffer) PyBuffer_Release(&buffer);
    Py_XDECREF(result);
    return NULL;
}

static PyObject *read_flat_path(PyObject *, PyObject *args, PyObject *kwargs)
{
    int file_id;
    const char *path;
    long long offset, count;
    const char *dtype = "uint8";
    static const char *names[] = {"file_id", "node_path", "flat_offset", "flat_count", "output_dtype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "isLL|s:read_flat_path",
        const_cast<char **>(names), &file_id, &path, &offset, &count, &dtype)) return NULL;
    try {
        // Both handles are obtained through the official capsule API.
        PdGDSFolder root = GDS_ID2FileRoot(file_id);
        PdGDSObj object = GDS_Node_Path(root, path, true);
        return read_array(object,offset,count,dtype);
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
    } catch (const char *error) {
        PyErr_SetString(PyExc_RuntimeError, error);
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "CoreArray file/node lookup failed");
    }
    return NULL;
}


static PyObject *read_selected_rows_path(PyObject *, PyObject *args, PyObject *kwargs)
{
    int file_id;
    const char *path;
    long long offset, rows, width;
    PyObject *selection;
    const char *dtype="uint8";
    static const char *names[]={"file_id","node_path","flat_offset","raw_rows","row_width","row_selection","output_dtype",NULL};
    if (!PyArg_ParseTupleAndKeywords(args,kwargs,"isLLLO|s:read_selected_rows_path",
        const_cast<char **>(names),&file_id,&path,&offset,&rows,&width,&selection,&dtype)) return NULL;
    if (offset < 0 || rows < 0 || width <= 0) {
        PyErr_SetString(PyExc_ValueError,"offset and raw_rows must be nonnegative and row_width positive");
        return NULL;
    }
    if (std::strcmp(dtype,"uint8") != 0) {
        PyErr_SetString(PyExc_ValueError,"raw Bit2 reads support output_dtype='uint8'");
        return NULL;
    }
    Py_buffer mask,output;
    bool have_mask=false,have_output=false;
    PyObject *result=NULL;
    try {
        PdGDSFolder root=GDS_ID2FileRoot(file_id);
        PdGDSObj object=GDS_Node_Path(root,path,true);
        C_Int64 total;
        PdAbstractArray array=bit2_array(object,&total);
        if (!array) return NULL;
        if (offset>total || rows>(total-offset)/width) {
            PyErr_SetString(PyExc_IndexError,"selected flat read exceeds the array element count");
            return NULL;
        }
        if (width>PY_SSIZE_T_MAX || rows>PY_SSIZE_T_MAX || offset%width!=0 || total%width!=0) {
            PyErr_SetString(PyExc_ValueError,"row_width must partition the flat array and offset must be row aligned");
            return NULL;
        }
        if (PyObject_GetBuffer(selection,&mask,PyBUF_C_CONTIGUOUS|PyBUF_FORMAT)<0) return NULL;
        have_mask=true;
        if (mask.ndim!=1 || mask.itemsize!=1 || mask.len!=width || !mask.format ||
            (std::strcmp(mask.format,"?")!=0 && std::strcmp(mask.format,"B")!=0 && std::strcmp(mask.format,"b")!=0)) {
            PyErr_SetString(PyExc_TypeError,"row_selection must be a contiguous bool or byte vector of row_width elements");
        } else {
            size_t selected=0;
            const unsigned char *bytes=static_cast<const unsigned char*>(mask.buf);
            for (long long i=0;i<width;++i) {
                if (bytes[i]>1) {
                    PyErr_SetString(PyExc_ValueError,"row_selection may only contain zero and one");
                    break;
                }
                selected+=bytes[i];
            }
            if (!PyErr_Occurred()) {
                if (selected && static_cast<unsigned long long>(rows)>static_cast<unsigned long long>(PY_SSIZE_T_MAX)/selected) {
                    PyErr_SetString(PyExc_OverflowError,"selected output exceeds Python buffer size");
                } else {
                    Py_ssize_t output_count=static_cast<Py_ssize_t>(rows)*static_cast<Py_ssize_t>(selected);
                    PyObject *numpy=PyImport_ImportModule("numpy");
                    if (numpy) {
                        result=PyObject_CallMethod(numpy,"empty","ns",output_count,"uint8");
                        Py_DECREF(numpy);
                    }
                    if (result && PyObject_GetBuffer(result,&output,PyBUF_WRITABLE|PyBUF_C_CONTIGUOUS)>=0) {
                        have_output=true;
                        if (output.len!=output_count || output.itemsize!=1) {
                            PyErr_SetString(PyExc_RuntimeError,"NumPy returned an unexpected uint8 buffer");
                        } else if (rows && selected) {
                            static_assert(sizeof(C_BOOL)==1,"the SDK boolean must be one byte");
                            CdIterator iterator;
                            GDS_Iter_Position(array,&iterator,static_cast<C_Int64>(offset));
                            unsigned char *destination=static_cast<unsigned char*>(output.buf);
                            for (long long row=0;row<rows;++row) {
                                void *end=GDS_Iter_RDataEx(&iterator,destination,static_cast<size_t>(width),svUInt8,
                                    reinterpret_cast<const C_BOOL*>(mask.buf));
                                destination+=selected;
                                if (end!=destination) {
                                    PyErr_SetString(PyExc_RuntimeError,"the SDK selected element count differs from the mask");
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError,error.what());
    } catch (const char *error) {
        PyErr_SetString(PyExc_RuntimeError,error);
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,"CoreArray selected iterator read failed");
    }
    if (have_output) PyBuffer_Release(&output);
    if (have_mask) PyBuffer_Release(&mask);
    if (PyErr_Occurred()) { Py_XDECREF(result); return NULL; }
    return result;
}

static PyMethodDef methods[] = {
    {"read_selected_rows_path", reinterpret_cast<PyCFunction>(read_selected_rows_path), METH_VARARGS | METH_KEYWORDS,
     "read_selected_rows_path(file_id, node_path, flat_offset, raw_rows, row_width, row_selection, output_dtype='uint8') -> selected raw Bit2 codes"},
    {"read_flat_path", reinterpret_cast<PyCFunction>(read_flat_path), METH_VARARGS | METH_KEYWORDS,
     "read_flat_path(file_id, node_path, flat_offset, flat_count, output_dtype='uint8') -> raw Bit2 codes"},
    {NULL, NULL, 0, NULL}
};
static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "staar_gds_flat", "Official PyGDS flat iterator I/O", -1, methods
};
PyMODINIT_FUNC PyInit_staar_gds_flat(void)
{
    if (Init_GDS_Routines() < 0) return NULL;
    return PyModule_Create(&module);
}
