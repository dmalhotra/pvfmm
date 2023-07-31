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


class FMMKernel(Enum):
    """Mirroring PVFMMKernel in pvfmm.h"""

    PVFMMLaplacePotential = 0
    PVFMMLaplaceGradient = 1
    PVFMMStokesPressure = 2
    PVFMMStokesVelocity = 3
    PVFMMStokesVelocityGrad = 4
    PVFMMBiotSavartPotential = 5


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
            ffi.PVFMMDestroyVolumeFMMD(ctypes.pointer(self._ptr))


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
            ffi.PVFMMDestroyContextD(ctypes.pointer(ctypes.c_void_p(self._ptr)))

    def evaluate(
        self,
        src_pos: np.ndarray,
        sl_den: np.ndarray,
        dl_den: Optional[np.ndarray],
        trg_pos: np.ndarray,
        setup: bool,
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
            src_pos, sl_den, dl_den, n_src, trg_pos, trg_val, n_trg, self._ptr, int(setup)
        )
        return trg_val
