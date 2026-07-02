import ctypes
import ctypes.util
import os

import numpy as np
from numpy.ctypeslib import ndpointer

# mpi4py is imported lazily (only when an explicit communicator is passed), so
# `import pvfmm` and the comm-less code paths work without an MPI installation.
MPI_Comm = ctypes.c_void_p  # provisional; refined by _ensure_mpi() on first use
_MPI = None


def _ensure_mpi():
    """Import mpi4py on first use and refine MPI_Comm (and the argtypes of the
    comm-taking entry points) to the platform's actual MPI_Comm width."""
    global _MPI, MPI_Comm
    if _MPI is None:
        # boilerplate from mpi4py/demo/wrap-ctypes/helloworld.py
        from mpi4py import MPI
        _MPI = MPI
        MPI_Comm = (
            ctypes.c_int
            if MPI._sizeof(MPI.Comm) == ctypes.sizeof(ctypes.c_int)
            else ctypes.c_void_p
        )
        for fn in _COMM_FUNCS:
            fn.argtypes = fn.argtypes[:-1] + [MPI_Comm]
    return _MPI


def get_MPI_COMM(comm):
    MPI = _ensure_mpi()
    return MPI_Comm.from_address(MPI._addressof(comm))


def wrapped_ndptr(*args, **kwargs):
    """
    A version of np.ctypeslib.ndpointer
    which allows None (passed as NULL)
    """
    base = ndpointer(*args, **kwargs)

    def from_param(cls, obj):
        if obj is None:
            return obj
        return base.from_param(obj)

    return type(base.__name__, (base,), {"from_param": classmethod(from_param)})


double_array = wrapped_ndptr(dtype=ctypes.c_double, flags=("C_CONTIGUOUS"))
float_array = wrapped_ndptr(dtype=ctypes.c_float, flags=("C_CONTIGUOUS"))


def volume_callback(type):
    return ctypes.CFUNCTYPE(
        None, ctypes.POINTER(type), ctypes.c_long, ctypes.POINTER(type), ctypes.c_void_p
    )


double_volume_callback = volume_callback(ctypes.c_double)
float_volume_callback = volume_callback(ctypes.c_float)

PVFMMKernel = ctypes.c_uint  # enum
PVFMMBoundaryType = ctypes.c_uint  # enum

# try to load lib
_custom_location = os.getenv("PVFMM")
if _custom_location:
    _dll = os.path.join(_custom_location, "libpvfmm.so")
else:
    _dll = ctypes.util.find_library("pvfmm")
    if _dll is None:
        raise ImportError(
            "Failed to find libpvfmm! \n"
            "Set PVFMM environment variable or check your installation of pvfmm!"
        )

SHARED_LIB = ctypes.CDLL(_dll)

# somwhat automatically generated:
# clang2py --clang-args="-I/mnt/sw/nix/store/z5w5a7pr5cmdbds0pn9ajdgy0jg71sl6-gcc-11.3.0/lib/gcc/x86_64-pc-linux-gnu/11.3.0/include/" include/pvfmm.h -l ~/pvfmm/build/libpvfmm.so > python/auto.py

PVFMMCreateVolumeFMMD = SHARED_LIB.PVFMMCreateVolumeFMMD
PVFMMCreateVolumeFMMD.restype = ctypes.c_void_p
PVFMMCreateVolumeFMMD.argtypes = [ctypes.c_int, ctypes.c_int, PVFMMKernel, MPI_Comm]

PVFMMCreateVolumeFMMF = SHARED_LIB.PVFMMCreateVolumeFMMF
PVFMMCreateVolumeFMMF.restype = ctypes.c_void_p
PVFMMCreateVolumeFMMF.argtypes = [ctypes.c_int, ctypes.c_int, PVFMMKernel, MPI_Comm]

PVFMMCreateVolumeTreeD = SHARED_LIB.PVFMMCreateVolumeTreeD
PVFMMCreateVolumeTreeD.restype = ctypes.c_void_p
PVFMMCreateVolumeTreeD.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    double_volume_callback,
    ctypes.c_void_p,
    double_array,
    ctypes.c_long,
    MPI_Comm,
    ctypes.c_double,
    ctypes.c_int,
    PVFMMBoundaryType,
    ctypes.c_int,
]
PVFMMCreateVolumeTreeF = SHARED_LIB.PVFMMCreateVolumeTreeF
PVFMMCreateVolumeTreeF.restype = ctypes.c_void_p
PVFMMCreateVolumeTreeF.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    float_volume_callback,
    ctypes.c_void_p,
    float_array,
    ctypes.c_long,
    MPI_Comm,
    ctypes.c_float,
    ctypes.c_int,
    PVFMMBoundaryType,
    ctypes.c_int,
]
PVFMMCreateVolumeTreeFromCoeffD = SHARED_LIB.PVFMMCreateVolumeTreeFromCoeffD
PVFMMCreateVolumeTreeFromCoeffD.restype = ctypes.c_void_p
PVFMMCreateVolumeTreeFromCoeffD.argtypes = [
    ctypes.c_long,
    ctypes.c_int,
    ctypes.c_int,
    double_array,
    double_array,
    double_array,
    ctypes.c_long,
    MPI_Comm,
    PVFMMBoundaryType,
]
PVFMMCreateVolumeTreeFromCoeffF = SHARED_LIB.PVFMMCreateVolumeTreeFromCoeffF
PVFMMCreateVolumeTreeFromCoeffF.restype = ctypes.c_void_p
PVFMMCreateVolumeTreeFromCoeffF.argtypes = [
    ctypes.c_long,
    ctypes.c_int,
    ctypes.c_int,
    float_array,
    float_array,
    float_array,
    ctypes.c_long,
    MPI_Comm,
    PVFMMBoundaryType,
]
PVFMMEvaluateVolumeFMMD = SHARED_LIB.PVFMMEvaluateVolumeFMMD
PVFMMEvaluateVolumeFMMD.restype = None
PVFMMEvaluateVolumeFMMD.argtypes = [
    double_array,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_long,
]
PVFMMEvaluateVolumeFMMF = SHARED_LIB.PVFMMEvaluateVolumeFMMF
PVFMMEvaluateVolumeFMMF.restype = None
PVFMMEvaluateVolumeFMMF.argtypes = [
    float_array,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_long,
]
PVFMMDestroyVolumeFMMD = SHARED_LIB.PVFMMDestroyVolumeFMMD
PVFMMDestroyVolumeFMMD.restype = None
PVFMMDestroyVolumeFMMD.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
PVFMMDestroyVolumeFMMF = SHARED_LIB.PVFMMDestroyVolumeFMMF
PVFMMDestroyVolumeFMMF.restype = None
PVFMMDestroyVolumeFMMF.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
PVFMMDestroyVolumeTreeD = SHARED_LIB.PVFMMDestroyVolumeTreeD
PVFMMDestroyVolumeTreeD.restype = None
PVFMMDestroyVolumeTreeD.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
PVFMMDestroyVolumeTreeF = SHARED_LIB.PVFMMDestroyVolumeTreeF
PVFMMDestroyVolumeTreeF.restype = None
PVFMMDestroyVolumeTreeF.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
PVFMMGetLeafCountD = SHARED_LIB.PVFMMGetLeafCountD
PVFMMGetLeafCountD.restype = ctypes.c_long
PVFMMGetLeafCountD.argtypes = [ctypes.c_void_p]
PVFMMGetLeafCountF = SHARED_LIB.PVFMMGetLeafCountF
PVFMMGetLeafCountF.restype = ctypes.c_long
PVFMMGetLeafCountF.argtypes = [ctypes.c_void_p]
PVFMMGetLeafCoordD = SHARED_LIB.PVFMMGetLeafCoordD
PVFMMGetLeafCoordD.restype = None
PVFMMGetLeafCoordD.argtypes = [double_array, ctypes.c_void_p]
PVFMMGetLeafCoordF = SHARED_LIB.PVFMMGetLeafCoordF
PVFMMGetLeafCoordF.restype = None
PVFMMGetLeafCoordF.argtypes = [float_array, ctypes.c_void_p]
PVFMMGetPotentialCoeffD = SHARED_LIB.PVFMMGetPotentialCoeffD
PVFMMGetPotentialCoeffD.restype = None
PVFMMGetPotentialCoeffD.argtypes = [
    double_array,
    ctypes.c_void_p,
]
PVFMMGetPotentialCoeffF = SHARED_LIB.PVFMMGetPotentialCoeffF
PVFMMGetPotentialCoeffF.restype = None
PVFMMGetPotentialCoeffF.argtypes = [
    float_array,
    ctypes.c_void_p,
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
PVFMMCreateContextD.restype = ctypes.c_void_p
PVFMMCreateContextD.argtypes = [
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int,
    PVFMMKernel,
    PVFMMBoundaryType,
    MPI_Comm,
]
PVFMMCreateContextF = SHARED_LIB.PVFMMCreateContextF
PVFMMCreateContextF.restype = ctypes.c_void_p
PVFMMCreateContextF.argtypes = [
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int,
    PVFMMKernel,
    PVFMMBoundaryType,
    MPI_Comm,
]

# World communicator as a Fortran integer handle (MPI_Fint, assumed int, as in
# PVFMM's Fortran interface); used by the comm-less path below.
PVFMMGetCommWorld = SHARED_LIB.PVFMMGetCommWorld
PVFMMGetCommWorld.restype = ctypes.c_int
PVFMMGetCommWorld.argtypes = []

# Fortran entry points for the particle context (all arguments by reference);
# these accept the Fortran integer comm handle and call MPI_Comm_f2c internally,
# so no knowledge of the platform MPI_Comm representation is needed.
_p_int = ctypes.POINTER(ctypes.c_int)
pvfmmcreatecontextd_ = SHARED_LIB.pvfmmcreatecontextd_
pvfmmcreatecontextd_.restype = None
pvfmmcreatecontextd_.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_double), _p_int, _p_int, _p_int, _p_int, _p_int]
pvfmmcreatecontextf_ = SHARED_LIB.pvfmmcreatecontextf_
pvfmmcreatecontextf_.restype = None
pvfmmcreatecontextf_.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_float), _p_int, _p_int, _p_int, _p_int, _p_int]

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
    ctypes.c_void_p,
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
    ctypes.c_void_p,
    ctypes.c_int,
]
PVFMMDestroyContextD = SHARED_LIB.PVFMMDestroyContextD
PVFMMDestroyContextD.restype = None
PVFMMDestroyContextD.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
PVFMMDestroyContextF = SHARED_LIB.PVFMMDestroyContextF
PVFMMDestroyContextF.restype = None
PVFMMDestroyContextF.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

# Entry points whose last argument is an MPI_Comm; _ensure_mpi() refines their
# argtypes once the real MPI_Comm width is known.
_COMM_FUNCS = (
    PVFMMCreateVolumeFMMD, PVFMMCreateVolumeFMMF,
    PVFMMCreateVolumeTreeD, PVFMMCreateVolumeTreeF,
    PVFMMCreateVolumeTreeFromCoeffD, PVFMMCreateVolumeTreeFromCoeffF,
    PVFMMCreateContextD, PVFMMCreateContextF,
)
