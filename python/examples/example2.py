# based on examples/src/example2-c.c

import numpy as np
import numba
from numba import types
import ctypes
from mpi4py import MPI

import pvfmm


@numba.njit
def fn_input(coord, out):
    n = len(coord) // 3
    L = 125
    for i in range(n):
        idx = i * 3
        c = coord[idx : idx + 3]
        r_2 = (c[0] - 0.5) ** 2 + (c[1] - 0.5) ** 2 + (c[2] - 0.5) ** 2
        out[idx + 0] = 0 + 2 * L * np.exp(-L * r_2) * (c[0] - 0.5)
        out[idx + 1] = 4 * L**2 * (c[2] - 0.5) * (5 - 2 * L * r_2) * np.exp(
            -L * r_2
        ) + 2 * L * np.exp(-L * r_2) * (c[1] - 0.5)
        out[idx + 2] = -4 * L**2 * (c[1] - 0.5) * (5 - 2 * L * r_2) * np.exp(
            -L * r_2
        ) + 2 * L * np.exp(-L * r_2) * (c[2] - 0.5)

c_sig = types.void(
    types.CPointer(types.double),
    types.int64,
    types.CPointer(types.double),
    types.voidptr,
)

# see https://numba.readthedocs.io/en/stable/user/cfunc.html
@numba.cfunc(c_sig, nopython=True)
def fn_input_C(coord_, n, out_, _ctx):
    coord = numba.carray(coord_, n * 3)
    out = numba.carray(out_, n * 3)
    fn_input(coord, out)


@numba.njit
def fn_poten(coord):
    n = len(coord) // 3
    dof = 3
    out = np.zeros(n * dof, dtype=np.float64)
    L = 125

    for i in range(n):
        idx = i * dof
        c = coord[idx : idx + dof]
        r_2 = (c[0] - 0.5) ** 2 + (c[1] - 0.5) ** 2 + (c[2] - 0.5) ** 2
        out[idx + 0] = 0
        out[idx + 1] = 2 * L * (c[2] - 0.5) * np.exp(-L * r_2)
        out[idx + 2] = -2 * L * (c[1] - 0.5) * np.exp(-L * r_2)

    return out


@numba.njit
def GetChebNodes(Nleaf, cheb_deg, depth, leaf_coord):
    leaf_length = 1.0 / (1 << depth)
    Ncheb = (cheb_deg + 1) ** 3
    cheb_coord = np.zeros(Nleaf * Ncheb * 3, dtype=np.float64)

    for leaf_idx in range(Nleaf):
        for j2 in range(cheb_deg + 1):
            for j1 in range(cheb_deg + 1):
                for j0 in range(cheb_deg + 1):
                    node_idx = (
                        leaf_idx * Ncheb
                        + (j2 * (cheb_deg + 1) + j1) * (cheb_deg + 1)
                        + j0
                    )
                    cheb_coord[node_idx * 3 + 0] = (
                        leaf_coord[leaf_idx * 3 + 0]
                        + (1 - np.cos(np.pi * (j0 * 2 + 1) / (cheb_deg * 2 + 2)))
                        * leaf_length
                        * 0.5
                    )
                    cheb_coord[node_idx * 3 + 1] = (
                        leaf_coord[leaf_idx * 3 + 1]
                        + (1 - np.cos(np.pi * (j1 * 2 + 1) / (cheb_deg * 2 + 2)))
                        * leaf_length
                        * 0.5
                    )
                    cheb_coord[node_idx * 3 + 2] = (
                        leaf_coord[leaf_idx * 3 + 2]
                        + (1 - np.cos(np.pi * (j2 * 2 + 1) / (cheb_deg * 2 + 2)))
                        * leaf_length
                        * 0.5
                    )

    return cheb_coord


def test1(fmm, kdim0, kdim1, cheb_deg, comm):
    Nt = 100
    trg_coord = np.random.rand(Nt * 3)
    trg_value_ref = fn_poten(trg_coord)
    tree = pvfmm.FMMVolumeTree.from_function(
        cheb_deg, kdim0, fn_input_C.ctypes, None, trg_coord, comm, 1e-6, 100, False, 0
    )
    trg_value = tree.evaluate(fmm, Nt)

    max_val = np.max(np.abs(trg_value_ref))
    max_err = np.max(np.abs(trg_value - trg_value_ref))
    print("Maximum relative error = ", max_err / max_val)


def test2(fmm, kdim0, kdim1, cheb_deg, comm):
    Ncheb = (cheb_deg + 1) * (cheb_deg + 1) * (cheb_deg + 1)
    Ncoef = (cheb_deg + 1) * (cheb_deg + 2) * (cheb_deg + 3) // 6

    depth = 3
    Nleaf = 1 << (3 * depth)
    leaf_length = 1 / (1 << depth)

    leaf_coord = np.empty(Nleaf * 3, dtype=np.float64)
    dens_value = np.empty(Nleaf * Ncheb * kdim0, dtype=np.float64)

    for leaf_idx in range(Nleaf):
        leaf_coord[leaf_idx * 3 + 0] = (
            (leaf_idx // (1 << (depth * 0))) % (1 << depth) * leaf_length
        )
        leaf_coord[leaf_idx * 3 + 1] = (
            (leaf_idx // (1 << (depth * 1))) % (1 << depth) * leaf_length
        )
        leaf_coord[leaf_idx * 3 + 2] = (
            (leaf_idx // (1 << (depth * 2))) % (1 << depth) * leaf_length
        )
    print("Getting nodes")
    cheb_coord = GetChebNodes(Nleaf, cheb_deg, depth, leaf_coord)

    fn_input(cheb_coord, dens_value)

    dense_coeff = pvfmm.nodes_to_coeff(Nleaf, cheb_deg, kdim0, dens_value)
    tree = pvfmm.FMMVolumeTree.from_coefficients(
        cheb_deg, kdim0, leaf_coord, dense_coeff, None, comm, False
    )
    tree.evaluate(fmm, 0)
    potn_value = tree.get_values()

    Nleaf = tree.leaf_count()  # TODO: does this change from above?
    leaf_coord = tree.get_leaf_coordinates()
    cheb_coord = GetChebNodes(Nleaf, cheb_deg, depth, leaf_coord)
    potn_value_ref = fn_poten(cheb_coord)

    max_val = np.max(np.abs(potn_value_ref))
    max_err = np.max(np.abs(potn_value - potn_value_ref))
    print("Maximum relative error = ", max_err / max_val)


if __name__ == "__main__":
    mult_order = 10
    cheb_deg = 14
    kdim0 = 3
    kdim1 = 3

    comm = MPI.COMM_WORLD
    fmm = pvfmm.FMMVolumeContext(
        mult_order, cheb_deg, pvfmm.FMMKernel.StokesVelocity, comm
    )
    test1(fmm, kdim0, kdim1, cheb_deg, comm)
    test2(fmm, kdim0, kdim1, cheb_deg, comm)
