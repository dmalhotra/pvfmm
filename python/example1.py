# module load gcc/11.3.0 openmpi python-mpi/3.10.8 fftw/mpi openblas
# export OMP_NUM_THREADS=16
# mpirun -n 1 --map-by slot:pe=$OMP_NUM_THREADS python -m mpi4py python/example1.py

# based on examples/src/example-c.c

import time

from mpi4py import MPI
import numpy as np
import numba

import pvfmm

@numba.njit(parallel=True)
def BiotSavart(src_X, src_V, trg_X):
    Ns = len(src_X) // 3
    Nt = len(trg_X) // 3
    trg_V = np.zeros(Nt * 3)
    oofp = 1 / (4 * np.pi)
    for t in numba.prange(Nt):
        for s in range(Ns):
            X0 = trg_X[t * 3 + 0] - src_X[s * 3 + 0]
            X1 = trg_X[t * 3 + 1] - src_X[s * 3 + 1]
            X2 = trg_X[t * 3 + 2] - src_X[s * 3 + 2]
            r2 = X0 * X0 + X1 * X1 + X2 * X2
            rinv = 1 / np.sqrt(r2) if r2 > 0 else 0
            rinv3 = rinv * rinv * rinv

            trg_V[t * 3 + 0] += (
                (src_V[s * 3 + 1] * X2 - src_V[s * 3 + 2] * X1) * rinv3 * oofp
            )
            trg_V[t * 3 + 1] += (
                (src_V[s * 3 + 2] * X0 - src_V[s * 3 + 0] * X2) * rinv3 * oofp
            )
            trg_V[t * 3 + 2] += (
                (src_V[s * 3 + 0] * X1 - src_V[s * 3 + 1] * X0) * rinv3 * oofp
            )
    return trg_V


def test_FMM(ctx):
    kdim = (3, 3)
    Ns = 20000
    Nt = 20000

    # Initialize target coordinates
    trg_X = np.random.rand(Nt * 3)

    # Initialize source coordinates and density
    src_X = np.random.rand(Ns * 3)
    src_V = np.random.rand(Ns * kdim[0]) - 0.5

    # FMM evaluation
    setup = 1
    tick = time.perf_counter_ns()
    trg_V = ctx.evaluate(src_X, src_V, None, trg_X, setup)
    tock = time.perf_counter_ns()
    print("FMM evaluation time (with setup) :", (tock - tick) / 1e9)

    # FMM evaluation (without setup)
    setup = 0
    tick = time.perf_counter_ns()
    trg_V2 = ctx.evaluate(src_X, src_V, None, trg_X, setup)
    tock = time.perf_counter_ns()
    print("FMM evaluation time (without setup) :", (tock - tick) / 1e9)

    # Direct evaluation
    tick = time.perf_counter_ns()
    trg_V0 = BiotSavart(src_X, src_V, trg_X)
    tock = time.perf_counter_ns()
    print("Direct evaluation time :", (tock - tick) / 1e9)

    max_val = np.max(np.abs(trg_V0))
    max_err = np.max(np.abs(trg_V - trg_V0))
    print("Max relative error :", max_err / max_val)

if __name__ == "__main__":
    # MPI init handled by mpi4py import
    box_size = -1
    points_per_box = 1000
    multipole_order = 10
    kernel = pvfmm.FMMKernel.PVFMMBiotSavartPotential

    print("Loaded.")
    ctx = pvfmm.FFMDoubleParticleContext(
        box_size, points_per_box, multipole_order, kernel, MPI.COMM_WORLD
    )
    print("Running!")
    test_FMM(ctx)

    # MPI finalize handled by mpi4py atexit handler
