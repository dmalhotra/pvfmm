# module load gcc/11.3.0 openmpi python/3.10.8
# source ~/venvs/pvfmm/bin/activate

import ctypes
import numpy as np
from mpi4py import MPI
from enum import Enum
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

# SLOW
# fmm = FMMContext(10,14, FMMKernel.PVFMMStokesVelocity, MPI.COMM_WORLD)

class FFMDoubleParticleContext:
    def __init__(
        self,
        box_size: int,
        max_points: int,
        multipole_order: int,
        kernel: FMMKernel,
        comm: MPI.Comm,
    ):
        if multipole_order <= 0 or multipole_order % 2 != 0:
            raise ValueError("multipole order must be even and postive")
        self._ptr = ffi.PVFMMCreateContextD(box_size,
            multipole_order, max_points, int(kernel.value), ffi.get_MPI_COMM(comm)
        )

    def __del__(self):
        if hasattr(self, "_ptr"):
            ffi.PVFMMDestroyContextD(ctypes.pointer(self._ptr))

# fmm = FFMDoubleParticleContext(-1, 1000,10,FMMKernel.PVFMMBiotSavartPotential, MPI.COMM_WORLD)
