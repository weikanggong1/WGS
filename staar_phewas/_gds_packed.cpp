// Optional pinned read-only allocator adapter; includes official headers, no vendored source.
// SPDX-License-Identifier: GPL-3.0-only
#define PY_SSIZE_T_CLEAN
#define COREARRAY_PYGDS_PACKAGE
#include <Python.h>
#include <PyGDS2.h>
#include "packed_binding.hpp"
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <exception>

static_assert(sizeof(CoreArray::SIZE64)==8, "CoreArray requires signed 64-bit positions");
static_assert(sizeof(C_Int64)==8 && sizeof(void*)==8 && sizeof(ssize_t)==8, "Packed adapter requires a 64-bit platform");

static bool runtime_binary_matches() {
    PyObject *m=PyImport_ImportModule("pygds.ccall");
    if (!m) return false;
    PyObject *name=PyObject_GetAttrString(m,"__file__");Py_DECREF(m);
    if (!name) return false;
    const char *path=PyUnicode_AsUTF8(name);
    if (!path) {Py_DECREF(name);return false;}
    std::ifstream f(path,std::ios::binary);
    std::string data((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>());
    Py_DECREF(name);
    if (data.empty()) {PyErr_SetString(PyExc_RuntimeError,"SDK binary unreadable");return false;}
    PyObject *h=PyImport_ImportModule("hashlib");if (!h)return false;
    PyObject *digest=PyObject_CallMethod(h,"sha256","y#",data.data(),static_cast<Py_ssize_t>(data.size()));Py_DECREF(h);
    if (!digest)return false;
    PyObject *hex=PyObject_CallMethod(digest,"hexdigest",NULL);Py_DECREF(digest);
    if (!hex)return false;
    const char *value=PyUnicode_AsUTF8(hex);
    bool ok=value && std::strcmp(value,BOUND_BINARY_SHA256)==0;Py_DECREF(hex);
    if (!ok && !PyErr_Occurred())PyErr_SetString(PyExc_RuntimeError,"Packed reader SDK binary binding mismatch");
    return ok;
}

static PyObject *binding(PyObject*,PyObject*) {
    return Py_BuildValue("{s:s,s:s,s:{s:n,s:n,s:n,s:n}}", "sdk_binary_sha256",BOUND_BINARY_SHA256,
        "official_headers_sha256",BOUND_HEADERS_SHA256,"layout",
        "iterator",sizeof(CoreArray::CdIterator),"allocator",sizeof(CoreArray::CdAllocator),
        "pointer",sizeof(void*),"position",sizeof(CoreArray::SIZE64));
}

// Stream bounds policy: -1 is the official decompressor's unknown-size sentinel.
static bool check_stream_bounds(CoreArray::SIZE64 signed_size,
                                unsigned long long logical_bytes,
                                unsigned long long first,
                                unsigned long long size) {
    if(signed_size < -1){
        PyErr_SetString(PyExc_RuntimeError,"Invalid negative allocator stream size");return false;
    }
    // These bounds come from the already-validated array element count.
    if(first > logical_bytes || size > logical_bytes-first){
        PyErr_SetString(PyExc_IndexError,"Packed byte range exceeds logical array");return false;
    }
    if(signed_size >= 0){
        unsigned long long known_size=static_cast<unsigned long long>(signed_size);
        if(known_size < logical_bytes || first > known_size || size > known_size-first){
            PyErr_SetString(PyExc_IndexError,"Packed allocator stream shorter than required range");return false;
        }
    }
    // Unknown physical length is enforced by CoreArray ReadData: short reads
    // and decompression failures throw, including EOF. Never substitute a size.
    return true;
}
// End stream bounds policy.

static PyObject *read_packed(PyObject*,PyObject *args) {
    int fileid;const char *path;long long offset,count,expected_samples;
    if (!PyArg_ParseTuple(args,"isLLL",&fileid,&path,&offset,&count,&expected_samples))return NULL;
    if(offset<0 || count<0){PyErr_SetString(PyExc_ValueError,"Negative packed range");return NULL;}
    PyObject *result=NULL;Py_buffer buffer;bool acquired=false;
    CoreArray::CdAllocator *allocator=NULL;CoreArray::SIZE64 old_position=0;bool restore=false;
    try {
        // Hold the GIL and owner through the official lookup and byte read.
        PdGDSObj node=GDS_Node_Path(GDS_ID2FileRoot(fileid),path,true);
        char cls[64]={};GDS_Node_GetClassName(node,cls,sizeof(cls)-1);
        if(std::strcmp(cls,"dBit2")!=0){PyErr_SetString(PyExc_TypeError,"Packed reader requires unsigned Bit2");return NULL;}
        PdAbstractArray array=static_cast<PdAbstractArray>(node);
        if(GDS_Array_GetBitOf(array)!=2 || !COREARRAY_SV_INTEGER(GDS_Array_GetSVType(array))) {
            PyErr_SetString(PyExc_TypeError,"Packed reader requires unsigned Bit2");return NULL;
        }
        int ndim=GDS_Array_DimCnt(array);C_Int32 dim[3];
        if(ndim!=3){PyErr_SetString(PyExc_ValueError,"Packed genotype must have three axes");return NULL;}
        GDS_Array_GetDim(array,dim,3);
        if(dim[0]<0 || dim[1]!=expected_samples || dim[2]!=2 || expected_samples<=0){
            PyErr_SetString(PyExc_ValueError,"Packed genotype axes/sample/ploidy mismatch");return NULL;
        }
        C_Int64 total=GDS_Array_GetTotalCount(array);
        const unsigned long long width=static_cast<unsigned long long>(dim[1])*dim[2];
        if(total<0 || width*static_cast<unsigned long long>(dim[0])!=static_cast<unsigned long long>(total)){
            PyErr_SetString(PyExc_ValueError,"Packed dimensions differ from total elements");return NULL;
        }
        if(offset>total || count>total-offset){PyErr_SetString(PyExc_IndexError,"Packed range exceeds array");return NULL;}
        unsigned long long first=static_cast<unsigned long long>(offset)>>2;
        unsigned long long within=static_cast<unsigned long long>(offset)&3;
        unsigned long long size=count ? (within+static_cast<unsigned long long>(count)+3)>>2 : 0;
        if(size>static_cast<unsigned long long>(PY_SSIZE_T_MAX)){PyErr_SetString(PyExc_OverflowError,"Packed Python buffer overflow");return NULL;}
        CoreArray::CdIterator iterator;GDS_Iter_Position(array,&iterator,offset);
        if(iterator.Ptr!=offset || !iterator.Allocator){PyErr_SetString(PyExc_RuntimeError,"Packed iterator binding mismatch");return NULL;}
        allocator=iterator.Allocator;
        CoreArray::SIZE64 signed_stream_size=allocator->GetSize();
        unsigned long long minimum_size=(static_cast<unsigned long long>(total)+3)>>2;
        if(!check_stream_bounds(signed_stream_size,minimum_size,first,size))return NULL;
        PyObject *numpy=PyImport_ImportModule("numpy");if(!numpy)return NULL;
        result=PyObject_CallMethod(numpy,"empty","ns",static_cast<Py_ssize_t>(size),"uint8");Py_DECREF(numpy);
        if(!result)return NULL;
        if(PyObject_GetBuffer(result,&buffer,PyBUF_WRITABLE|PyBUF_C_CONTIGUOUS)<0){Py_DECREF(result);return NULL;}
        acquired=true;
        if(buffer.len!=static_cast<Py_ssize_t>(size) || buffer.itemsize!=1)throw "Unexpected packed output buffer";
        if(size){
            old_position=allocator->Position();restore=true;
            allocator->SetPosition(first);
            allocator->ReadData(buffer.buf,static_cast<ssize_t>(size));
            allocator->SetPosition(old_position);restore=false;
        }
        PyBuffer_Release(&buffer);acquired=false;
        return Py_BuildValue("(NLLLL)",result,static_cast<long long>(within*2),count,
                             static_cast<long long>(first),static_cast<long long>(signed_stream_size));
    }catch(const std::exception &e){PyErr_SetString(PyExc_RuntimeError,e.what());}
     catch(const char *e){PyErr_SetString(PyExc_RuntimeError,e);}
     catch(...){PyErr_SetString(PyExc_RuntimeError,"Packed allocator read failed");}
    if(restore){try{allocator->SetPosition(old_position);}catch(...){
        PyErr_SetString(PyExc_RuntimeError,"Packed allocator position restore failed; close and reopen the reader");
    }}
    if(acquired)PyBuffer_Release(&buffer);Py_XDECREF(result);return NULL;
}
static PyMethodDef methods[]={
 {"binding",binding,METH_NOARGS,"Pinned SDK/header and compiler layout identities"},
 {"read_packed_path",read_packed,METH_VARARGS,"Packed Bit2 range plus bit offset/count/stream metadata"},
 {NULL,NULL,0,NULL}};
static PyModuleDef module={PyModuleDef_HEAD_INIT,"staar_gds_packed","Pinned CoreArray logical byte reader",-1,methods};
PyMODINIT_FUNC PyInit_staar_gds_packed(){
 if(!runtime_binary_matches() || Init_GDS_Routines()<0)return NULL;
 return PyModule_Create(&module);
}
