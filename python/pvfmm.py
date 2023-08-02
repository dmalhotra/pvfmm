# module load gcc/11.3.0 openmpi python/3.10.8 openblas fftw3
# source ~/venvs/pvfmm/bin/activate

import ctypes
import numpy as np
from enum import Enum
from typing import Optional

# calls MPI_Init_thread() and sets up MPI_Finalize() for you.
# see https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html
from mpi4py import MPI

import ffi


def nodes_to_coeff(N_leaf: int, cheb_deg: int, dof: int, node_val: np.ndarray):
    is_double = node_val.dtype == np.float64
    Ncoef = (cheb_deg + 1) * (cheb_deg + 2) * (cheb_deg + 3) // 6
    # XXX: is this valid?
    coeff = np.empty(Ncoef * N_leaf * dof, dtype=node_val.dtype)
    if is_double:
        ffi.PVFMMNodes2CoeffD(coeff, N_leaf, cheb_deg, dof, node_val)
    else:
        ffi.PVFMMNodes2CoeffF(coeff, N_leaf, cheb_deg, dof, node_val)

    return coeff


class FMMKernel(Enum):
    """Mirroring PVFMMKernel in pvfmm.h"""

    LaplacePotential = 0
    LaplaceGradient = 1
    StokesPressure = 2
    StokesVelocity = 3
    StokesVelocityGrad = 4
    BiotSavartPotential = 5


class FMMDoubleVolumeContext:
    def __init__(
        self,
        multipole_order: int,
        chebyshev_degree: int,
        kernel: FMMKernel,
        comm: MPI.Comm,
    ):
        if multipole_order <= 0 or multipole_order % 2 != 0:
            raise ValueError("multipole order must be even and postive")
        self._ptr = ffi.PVFMMCreateVolumeFMMD(
            multipole_order, chebyshev_degree, int(kernel.value), ffi.get_MPI_COMM(comm)
        )

    def __del__(self):
        if hasattr(self, "_ptr"):
            ffi.PVFMMDestroyVolumeFMMD(ctypes.byref(ctypes.c_void_p(self._ptr)))


class FFMDoubleParticleContext:
    def __init__(
        self,
        box_size: float,
        max_points: int,
        multipole_order: int,
        kernel: FMMKernel,
        comm: MPI.Comm,
    ):
        if multipole_order <= 0 or multipole_order % 2 != 0:
            raise ValueError("multipole order must be even and postive")
        self._ptr = ffi.PVFMMCreateContextD(
            float(box_size),
            max_points,
            multipole_order,
            int(kernel.value),
            ffi.get_MPI_COMM(comm),
        )

    def __del__(self):
        if hasattr(self, "_ptr"):
            ffi.PVFMMDestroyContextD(ctypes.byref(ctypes.c_void_p(self._ptr)))

    def evaluate(
        self,
        src_pos: np.ndarray,
        sl_den: np.ndarray,
        dl_den: Optional[np.ndarray],
        trg_pos: np.ndarray,
        setup: bool = True,
    ):
        source_length = len(src_pos)
        if source_length % 3 != 0:
            raise ValueError(
                "Source arrays must have a length which is a multiple of 3"
            )
        if dl_den is not None and len(sl_den) != source_length:
            raise ValueError("Source arrays must all be of the same length!")
        if dl_den is not None and len(dl_den) != source_length * 2:
            raise ValueError("Source arrays must all be of the same length!")
        n_src = source_length // 3

        target_length = len(trg_pos)
        if target_length % 3 != 0:
            raise ValueError(
                "Target arrays must have a length which is a multiple of 3"
            )
        n_trg = target_length // 3
        trg_val = np.empty(target_length)

        ffi.PVFMMEvalD(
            src_pos,
            sl_den,
            dl_den,
            n_src,
            trg_pos,
            trg_val,
            n_trg,
            self._ptr,
            int(setup),
        )
        return trg_val


class FMMDoubleVolumeTree:
    def __init__(self, ptr: ctypes.c_void_p, cheb_deg: int, data_dim: int, n_trg: int):
        self._ptr = ptr
        self.cheb_deg = cheb_deg
        self.n_cheb = (cheb_deg + 1) ** 3
        self.n_coeff = (cheb_deg + 1) * (cheb_deg + 2) * (cheb_deg + 3) // 6
        self.data_dim = data_dim
        self.n_trg = n_trg

    @classmethod
    def from_function(
        cls,
        cheb_deg: int,
        data_dim: int,
        fn: ffi.double_volume_callback,
        context: ctypes.c_void_p,  # TODO: investigate
        trg_coord: np.ndarray,
        comm: MPI.Comm,
        tol: float,
        max_pts: int,
        periodic: bool,
        init_depth: int,
    ):
        n_trg = len(trg_coord) // 3
        ptr = ffi.PVFMMCreateVolumeTreeD(
            cheb_deg,
            data_dim,
            fn,
            context,
            trg_coord,
            n_trg,
            ffi.get_MPI_COMM(comm),
            tol,
            max_pts,
            periodic,
            init_depth,
        )
        return cls(ptr, cheb_deg, data_dim, n_trg)

    @classmethod
    def from_coefficients(
        cls,
        cheb_deg: int,
        data_dim: int,
        leaf_coord: np.ndarray,
        fn_coeff: np.ndarray,
        trg_coord: Optional[np.ndarray],
        comm: MPI.Comm,
        periodic: bool,
    ):
        if len(leaf_coord) % 3 != 0:
            raise ValueError(
                "Leaf coordinates must have a length which is a multiple of 3"
            )
        N_leaf = len(leaf_coord) // 3
        fn_coeff_size = (
            N_leaf * data_dim * (cheb_deg + 1) * (cheb_deg + 2) * (cheb_deg + 3) // 6
        )
        if len(fn_coeff) != fn_coeff_size:
            raise ValueError(
                "Function coefficients array has the wrong length, required length "
                + fn_coeff_size
            )
        n_trg = len(trg_coord) // 3 if trg_coord is not None else 0

        ptr = ffi.PVFMMCreateVolumeTreeFromCoeffD(
            N_leaf,
            cheb_deg,
            data_dim,
            leaf_coord,
            fn_coeff,
            trg_coord,
            n_trg,
            ffi.get_MPI_COMM(comm),
            periodic,
        )
        return cls(ptr, cheb_deg, data_dim, n_trg)

    def __del__(self):
        if hasattr(self, "_ptr"):
            ffi.PVFMMDestroyVolumeTreeD(ctypes.byref(ctypes.c_void_p(self._ptr)))

    def evaluate(self, fmm: FMMDoubleVolumeContext, loc_size: int) -> np.ndarray:
        # TODO: XXX: in C code this uses kdim1, but data_dim was passed kdim0
        trg_val = np.empty(self.n_trg * self.data_dim)
        ffi.PVFMMEvaluateVolumeFMMD(trg_val, self._ptr, fmm._ptr, loc_size)
        self._evaluated = True
        return trg_val

    def leaf_count(self):
        return int(ffi.PVFMMGetLeafCountD(self._ptr))

    def get_leaf_coordinates(self):
        Nleaf = self.leaf_count()
        leaf_coord = np.empty(Nleaf * 3)
        ffi.PVFMMGetLeafCoordD(leaf_coord, self._ptr)
        return leaf_coord

    # WHERE DO I GET KDIM0?

    def get_coefficients(self):
        if not self._evaluated:
            raise ValueError(
                "Cannot get coefficients of an un-evaluated tree"
            )  # TODO: true?
        n_leaf = self.leaf_count()
        # TODO: XXX: in C code this uses kdim1, but data_dim was passed kdim0
        coeff = np.empty(n_leaf * self.n_coeff * self.data_dim)
        ffi.PVFMMGetPotentialCoeffD(coeff, self._ptr)
        return coeff

    def get_values(self):
        coeff = self.get_coefficients()
        n_leaf = self.leaf_count()
        # TODO: XXX: in C code these two lines use kdim1, but data_dim was passed kdim0
        value = np.empty(n_leaf * self.n_cheb * self.data_dim)
        ffi.PVFMMCoeff2NodesD(value, n_leaf, self.cheb_deg, self.data_dim, coeff)
        return value
