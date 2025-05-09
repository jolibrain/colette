import array

import numpy as np
from faiss.swigfaiss import memcpy, swig_ptr


def my_vector_to_array(v):
    sizeof_long = array.array("l").itemsize
    deprecated_name_map = {
        # deprecated: replacement
        "Float": "Float32",
        "Double": "Float64",
        "Char": "Int8",
        "Int": "Int32",
        "Long": "Int32" if sizeof_long == 4 else "Int64",
        "LongLong": "Int64",
        "Byte": "UInt8",
        # previously misspelled variant
        "Uint64": "UInt64",
    }

    """convert a C++ vector to a numpy array"""
    vector_name_map = {
        "Float32": "float32",
        "Float64": "float64",
        "Int8": "int8",
        "Int16": "int16",
        "Int32": "int32",
        "Int64": "int64",
        "UInt8": "uint8",
        "UInt16": "uint16",
        "UInt32": "uint32",
        "UInt64": "uint64",
        **{k: v.lower() for k, v in deprecated_name_map.items()},
    }

    classname = v.__class__.__name__
    assert (
        classname.endswith("Vector")
        or classname.endswith("vectorf")
        or classname.endswith("vectord")
    )
    if classname.endswith("vectorf"):
        dtype = np.dtype("float32")
    elif classname.endswith("vectorf"):
        dtype = np.dtype("float64")
    elif classname.endswith("Vector"):
        dtype = np.dtype(vector_name_map[classname[:-6]])
    a = np.empty(v.size(), dtype=dtype)
    if v.size() > 0:
        if classname.endswith("Vector"):
            memcpy(swig_ptr(a), v.data(), a.nbytes)
        else:
            for i, x in enumerate(v):
                a[i] = x
    return a
