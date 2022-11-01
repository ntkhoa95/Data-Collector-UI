import os
import ctypes
import numpy as np


_file = 'libhf.so'
_path = os.path.join(os.path.dirname(__file__), _file)
print(_path)
_mod = ctypes.cdll.LoadLibrary(_path)


class IntArrayType:
    def from_param(self, param):
        assert isinstance(param, np.ndarray), "Not numpy array!"
        assert param.dtype == np.int32, "Must be int32"
        return param.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


IntArray = IntArrayType()


class Answer(ctypes.Structure):
    _fields_ = [('nbBgRegions', ctypes.c_int),
                ('neighbor', ctypes.POINTER(ctypes.c_int)),
                ('visited', ctypes.POINTER(ctypes.c_int))]


_count_neighbor = _mod.count_neighbor
_count_neighbor.argtypes = (IntArray, ctypes.c_int, ctypes.c_int)
_count_neighbor.restype = ctypes.POINTER(Answer)


answer_free = _mod.answer_free
answer_free.argtypes = (ctypes.POINTER(Answer),)


def fill_bg_with_fg(mask):
    mask = mask.copy()
    if np.all(mask == 0):
        return mask
    h, w = mask.shape
    p = _count_neighbor(mask, h, w)
    ans = p.contents
    neighbor = np.frombuffer(ctypes.cast(ans.neighbor, ctypes.POINTER(ctypes.c_int * ans.nbBgRegions))[0], np.int32)
    visited = np.frombuffer(ctypes.cast(ans.visited, ctypes.POINTER(ctypes.c_int * (h * w)))[0], np.int32)
    visited = visited.reshape((h, w))
    for i, nb in enumerate(neighbor):
        if nb != -2:
            mask[visited == i] = nb
    answer_free(p)
    return mask
