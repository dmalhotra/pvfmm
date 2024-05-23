import ctypes

import numpy as np
from numpy.ctypeslib import ndpointer
from mpi4py import MPI

# boilerplate from https://github.com/mpi4py/mpi4py/blob/master/demo/wrap-ctypes/helloworld.py
if MPI._sizeof(MPI.Comm) == ctypes.sizeof(ctypes.c_int):
    MPI_Comm = ctypes.c_int
else:
    MPI_Comm = ctypes.c_void_p


def get_MPI_COMM(comm) -> MPI_Comm:
    comm_ptr = MPI._addressof(comm)
    return MPI_Comm.from_address(comm_ptr)

def wrapped_ndptr(*args, **kwargs):
  base = ndpointer(*args, **kwargs)
  def from_param(cls, obj):
    if obj is None:
      return obj
    return base.from_param(obj)
  return type(base.__name__, (base,), {'from_param': classmethod(from_param)})

double_array = wrapped_ndptr(dtype=ctypes.c_double, flags=("C_CONTIGUOUS"))
float_array = wrapped_ndptr(dtype=ctypes.c_float, flags=("C_CONTIGUOUS"))

PVFMMKernel = ctypes.c_uint  # enum

# try to load lib
# hardcoded path for now
SHARED_LIB = ctypes.CDLL("./build/libpvfmm.so")


# somwhat automatically generated:
# clang2py --clang-args="-I/mnt/sw/nix/store/z5w5a7pr5cmdbds0pn9ajdgy0jg71sl6-gcc-11.3.0/lib/gcc/x86_64-pc-linux-gnu/11.3.0/include/" include/pvfmm.h -l ~/pvfmm/build/libpvfmm.so > python/auto.py

PVFMMCreateVolumeFMMD = SHARED_LIB.PVFMMCreateVolumeFMMD
PVFMMCreateVolumeFMMD.restype = ctypes.POINTER(None)
PVFMMCreateVolumeFMMD.argtypes = [ctypes.c_int, ctypes.c_int, PVFMMKernel, MPI_Comm]

PVFMMCreateVolumeFMMF = SHARED_LIB.PVFMMCreateVolumeFMMF
PVFMMCreateVolumeFMMF.restype = ctypes.POINTER(None)
PVFMMCreateVolumeFMMF.argtypes = [ctypes.c_int, ctypes.c_int, PVFMMKernel, MPI_Comm]

PVFMMCreateVolumeTreeD = SHARED_LIB.PVFMMCreateVolumeTreeD
PVFMMCreateVolumeTreeD.restype = ctypes.POINTER(None)
PVFMMCreateVolumeTreeD.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.CFUNCTYPE(
        None,
        double_array,
        ctypes.c_long,
        double_array,
        ctypes.POINTER(None),
    ),
    ctypes.POINTER(None),
    double_array,
    ctypes.c_long,
    MPI_Comm,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_bool,
    ctypes.c_int,
]
PVFMMCreateVolumeTreeF = SHARED_LIB.PVFMMCreateVolumeTreeF
PVFMMCreateVolumeTreeF.restype = ctypes.POINTER(None)
PVFMMCreateVolumeTreeF.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.CFUNCTYPE(
        None,
        float_array,
        ctypes.c_long,
        float_array,
        ctypes.POINTER(None),
    ),
    ctypes.POINTER(None),
    float_array,
    ctypes.c_long,
    MPI_Comm,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_bool,
    ctypes.c_int,
]
PVFMMCreateVolumeTreeFromCoeffD = SHARED_LIB.PVFMMCreateVolumeTreeFromCoeffD
PVFMMCreateVolumeTreeFromCoeffD.restype = ctypes.POINTER(None)
PVFMMCreateVolumeTreeFromCoeffD.argtypes = [
    ctypes.c_long,
    ctypes.c_int,
    ctypes.c_int,
    double_array,
    double_array,
    double_array,
    ctypes.c_long,
    MPI_Comm,
    ctypes.c_bool,
]
PVFMMCreateVolumeTreeFromCoeffF = SHARED_LIB.PVFMMCreateVolumeTreeFromCoeffF
PVFMMCreateVolumeTreeFromCoeffF.restype = ctypes.POINTER(None)
PVFMMCreateVolumeTreeFromCoeffF.argtypes = [
    ctypes.c_long,
    ctypes.c_int,
    ctypes.c_int,
    float_array,
    float_array,
    float_array,
    ctypes.c_long,
    MPI_Comm,
    ctypes.c_bool,
]
PVFMMEvaluateVolumeFMMD = SHARED_LIB.PVFMMEvaluateVolumeFMMD
PVFMMEvaluateVolumeFMMD.restype = None
PVFMMEvaluateVolumeFMMD.argtypes = [
    double_array,
    ctypes.POINTER(None),
    ctypes.POINTER(None),
    ctypes.c_long,
]
PVFMMEvaluateVolumeFMMF = SHARED_LIB.PVFMMEvaluateVolumeFMMF
PVFMMEvaluateVolumeFMMF.restype = None
PVFMMEvaluateVolumeFMMF.argtypes = [
    float_array,
    ctypes.POINTER(None),
    ctypes.POINTER(None),
    ctypes.c_long,
]
PVFMMDestroyVolumeFMMD = SHARED_LIB.PVFMMDestroyVolumeFMMD
PVFMMDestroyVolumeFMMD.restype = None
PVFMMDestroyVolumeFMMD.argtypes = [ctypes.POINTER(ctypes.POINTER(None))]
PVFMMDestroyVolumeFMMF = SHARED_LIB.PVFMMDestroyVolumeFMMF
PVFMMDestroyVolumeFMMF.restype = None
PVFMMDestroyVolumeFMMF.argtypes = [ctypes.POINTER(ctypes.POINTER(None))]
PVFMMDestroyVolumeTreeD = SHARED_LIB.PVFMMDestroyVolumeTreeD
PVFMMDestroyVolumeTreeD.restype = None
PVFMMDestroyVolumeTreeD.argtypes = [ctypes.POINTER(ctypes.POINTER(None))]
PVFMMDestroyVolumeTreeF = SHARED_LIB.PVFMMDestroyVolumeTreeF
PVFMMDestroyVolumeTreeF.restype = None
PVFMMDestroyVolumeTreeF.argtypes = [ctypes.POINTER(ctypes.POINTER(None))]
PVFMMGetLeafCountD = SHARED_LIB.PVFMMGetLeafCountD
PVFMMGetLeafCountD.restype = ctypes.c_long
PVFMMGetLeafCountD.argtypes = [ctypes.POINTER(None)]
PVFMMGetLeafCountF = SHARED_LIB.PVFMMGetLeafCountF
PVFMMGetLeafCountF.restype = ctypes.c_long
PVFMMGetLeafCountF.argtypes = [ctypes.POINTER(None)]
PVFMMGetLeafCoordD = SHARED_LIB.PVFMMGetLeafCoordD
PVFMMGetLeafCoordD.restype = None
PVFMMGetLeafCoordD.argtypes = [double_array, ctypes.POINTER(None)]
PVFMMGetLeafCoordF = SHARED_LIB.PVFMMGetLeafCoordF
PVFMMGetLeafCoordF.restype = None
PVFMMGetLeafCoordF.argtypes = [float_array, ctypes.POINTER(None)]
PVFMMGetPotentialCoeffD = SHARED_LIB.PVFMMGetPotentialCoeffD
PVFMMGetPotentialCoeffD.restype = None
PVFMMGetPotentialCoeffD.argtypes = [
    double_array,
    ctypes.POINTER(None),
]
PVFMMGetPotentialCoeffF = SHARED_LIB.PVFMMGetPotentialCoeffF
PVFMMGetPotentialCoeffF.restype = None
PVFMMGetPotentialCoeffF.argtypes = [
    float_array,
    ctypes.POINTER(None),
]
PVFMMCoeff2NodesD = SHARED_LIB.PVFMMCoeff2NodesD
PVFMMCoeff2NodesD.restype = None
PVFMMCoeff2NodesD.argtypes = [
    double_array,
    ctypes.c_long,
    ctypes.c_int,
    ctypes.c_int,
    double_array,
]
PVFMMCoeff2NodesF = SHARED_LIB.PVFMMCoeff2NodesF
PVFMMCoeff2NodesF.restype = None
PVFMMCoeff2NodesF.argtypes = [
    float_array,
    ctypes.c_long,
    ctypes.c_int,
    ctypes.c_int,
    float_array,
]
PVFMMNodes2CoeffD = SHARED_LIB.PVFMMNodes2CoeffD
PVFMMNodes2CoeffD.restype = None
PVFMMNodes2CoeffD.argtypes = [
    double_array,
    ctypes.c_long,
    ctypes.c_int,
    ctypes.c_int,
    double_array,
]
PVFMMNodes2CoeffF = SHARED_LIB.PVFMMNodes2CoeffF
PVFMMNodes2CoeffF.restype = None
PVFMMNodes2CoeffF.argtypes = [
    float_array,
    ctypes.c_long,
    ctypes.c_int,
    ctypes.c_int,
    float_array,
]
PVFMMCreateContextD = SHARED_LIB.PVFMMCreateContextD
PVFMMCreateContextD.restype = ctypes.POINTER(None)
PVFMMCreateContextD.argtypes = [
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int,
    PVFMMKernel,
    MPI_Comm,
]
PVFMMCreateContextF = SHARED_LIB.PVFMMCreateContextF
PVFMMCreateContextF.restype = ctypes.POINTER(None)
PVFMMCreateContextF.argtypes = [
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int,
    PVFMMKernel,
    MPI_Comm,
]
PVFMMEvalD = SHARED_LIB.PVFMMEvalD
PVFMMEvalD.restype = None
PVFMMEvalD.argtypes = [
    double_array,
    double_array,
    double_array,
    ctypes.c_long,
    double_array,
    double_array,
    ctypes.c_long,
    ctypes.POINTER(None),
    ctypes.c_int,
]
PVFMMEvalF = SHARED_LIB.PVFMMEvalF
PVFMMEvalF.restype = None
PVFMMEvalF.argtypes = [
    float_array,
    float_array,
    float_array,
    ctypes.c_long,
    float_array,
    float_array,
    ctypes.c_long,
    ctypes.POINTER(None),
    ctypes.c_int,
]
PVFMMDestroyContextD = SHARED_LIB.PVFMMDestroyContextD
PVFMMDestroyContextD.restype = None
PVFMMDestroyContextD.argtypes = [ctypes.POINTER(ctypes.POINTER(None))]
PVFMMDestroyContextF = SHARED_LIB.PVFMMDestroyContextF
PVFMMDestroyContextF.restype = None
PVFMMDestroyContextF.argtypes = [ctypes.POINTER(ctypes.POINTER(None))]
