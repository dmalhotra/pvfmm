"""Pythonic layer over the PVFMM C API (raw ctypes bindings live in pvfmm.ffi)."""

from __future__ import annotations

import ctypes
import numpy as np
from enum import Enum
from typing import Optional, Callable, Union

from . import ffi


def nodes_to_coeff(
    N_leaf: int, cheb_deg: int, dof: int, node_val: np.ndarray
) -> np.ndarray:
    """Convert function values on tensor-product Chebyshev nodes (first kind)
    to Chebyshev coefficients.

    node_val holds N_leaf*(cheb_deg+1)^3*dof values; the result has
    N_leaf*(cheb_deg+1)(cheb_deg+2)(cheb_deg+3)/6*dof coefficients.
    """
    is_double = node_val.dtype == np.float64
    Ncoef = (cheb_deg + 1) * (cheb_deg + 2) * (cheb_deg + 3) // 6
    # TODO: is this the valid size of the output array?
    coeff = np.empty(Ncoef * N_leaf * dof, dtype=node_val.dtype)

    get_function_dtype("PVFMMNodes2Coeff", node_val.dtype)(
        coeff, N_leaf, cheb_deg, dof, node_val
    )

    return coeff


class FMMKernel(Enum):
    """Mirroring PVFMMKernel in pvfmm.h"""

    LaplacePotential = 0
    LaplaceGradient = 1
    StokesPressure = 2
    StokesVelocity = 3
    StokesVelocityGrad = 4
    BiotSavartPotential = 5


class FMMBoundaryType(Enum):
    """Mirroring PVFMMBoundaryType in pvfmm.h; values 0/1 coincide with the
    boolean periodic flag."""

    FreeSpace = 0
    PXYZ = 1
    PX = 2
    PXY = 3
    Periodic = 1  # alias for PXYZ


def _boundary_value(boundary: Union[bool, FMMBoundaryType]) -> int:
    if isinstance(boundary, FMMBoundaryType):
        return boundary.value
    return int(bool(boundary))


# read out of calls to BuildKernel in pvfmm/include/kernel.txx
KERNEL_DIMS = {
    FMMKernel.LaplacePotential: (1, 1),
    FMMKernel.LaplaceGradient: (1, 3),
    FMMKernel.StokesPressure: (3, 1),
    FMMKernel.StokesVelocity: (3, 3),
    FMMKernel.StokesVelocityGrad: (3, 9),
    FMMKernel.BiotSavartPotential: (3, 3),
}


def get_function_dtype(function_name: str, dtype: np.dtype) -> Callable:
    """
    Helper function to switch between the double and float functions
    from the FFI module
    """
    if dtype == np.float64:
        function_name += "D"
    elif dtype == np.float32:
        function_name += "F"
    else:
        raise ValueError("Invalid dtype, must be either float64 or float32")

    return getattr(ffi, function_name)


class FMMVolumeContext:
    """Volume-FMM translation operators for one (kernel, multipole_order,
    chebyshev_degree, dtype) combination.

    Construction precomputes (or loads from the Precomp_* cache; see
    $PVFMM_DIR) the operators, which can take a while on first use. Pass the
    instance to FMMVolumeTree.evaluate(). comm is an mpi4py communicator.
    """

    def __init__(
        self,
        multipole_order: int,
        chebyshev_degree: int,
        kernel: FMMKernel,
        comm: MPI.Comm,
        dtype=np.float64,
    ):
        self.kernel = kernel
        self.dtype = np.dtype(dtype)
        if multipole_order <= 0 or multipole_order % 2 != 0:
            raise ValueError("multipole order must be even and postive")
        self._ptr = get_function_dtype("PVFMMCreateVolumeFMM", dtype)(
            multipole_order,
            chebyshev_degree,
            int(self.kernel.value),
            ffi.get_MPI_COMM(comm),
        )

    def __del__(self):
        if hasattr(self, "_ptr"):
            get_function_dtype("PVFMMDestroyVolumeFMM", self.dtype)(
                ctypes.byref(ctypes.c_void_p(self._ptr))
            )


class FMMParticleContext:
    """Particle-FMM evaluator for one kernel.

    box_size is the domain length and the period along the periodic
    directions (must be > 0 for periodic boundaries; <= 0 with free space
    means the bounding box is computed from the points). boundary is an
    FMMBoundaryType; if None, box_size > 0 selects fully periodic and
    box_size <= 0 free space. comm is an mpi4py communicator; if None, the
    world communicator is obtained from the library and mpi4py is not needed.
    """

    def __init__(
        self,
        box_size: float,
        max_points: int,
        multipole_order: int,
        kernel: FMMKernel,
        comm=None,
        dtype=np.float64,
        boundary: Optional[FMMBoundaryType] = None,
    ):
        self.kernel = kernel
        self.dtype = np.dtype(dtype)
        if multipole_order <= 0 or multipole_order % 2 != 0:
            raise ValueError("multipole order must be even and postive")

        if boundary is None:
            # legacy convention: box_size <= 0 -> free space, > 0 -> fully periodic
            boundary = (
                FMMBoundaryType.Periodic if box_size > 0 else FMMBoundaryType.FreeSpace
            )
        if comm is None:
            # No communicator (and possibly no mpi4py): obtain the world
            # communicator from the library as an MPI_Fint and create the
            # context through the Fortran entry point, which does MPI_Comm_f2c.
            box_t = ctypes.c_double if self.dtype == np.float64 else ctypes.c_float
            shim = (
                ffi.pvfmmcreatecontextd_
                if self.dtype == np.float64
                else ffi.pvfmmcreatecontextf_
            )
            ctx = ctypes.c_void_p()
            fint = ctypes.c_int(ffi.PVFMMGetCommWorld())
            shim(
                ctypes.byref(ctx),
                ctypes.byref(box_t(box_size)),
                ctypes.byref(ctypes.c_int(max_points)),
                ctypes.byref(ctypes.c_int(multipole_order)),
                ctypes.byref(ctypes.c_int(int(self.kernel.value))),
                ctypes.byref(ctypes.c_int(int(_boundary_value(boundary)))),
                ctypes.byref(fint),
            )
            self._ptr = ctx.value
        else:
            self._ptr = get_function_dtype("PVFMMCreateContext", dtype)(
                float(box_size),
                max_points,
                multipole_order,
                int(self.kernel.value),
                _boundary_value(boundary),
                ffi.get_MPI_COMM(comm),
            )
        if self._ptr is None:
            raise ValueError(
                "PVFMMCreateContext failed (periodic boundaries need box_size > 0)"
            )

    def __del__(self):
        if hasattr(self, "_ptr"):
            get_function_dtype("PVFMMDestroyContext", self.dtype)(
                ctypes.byref(ctypes.c_void_p(self._ptr))
            )

    def evaluate(
        self,
        src_pos: np.ndarray,
        sl_den: np.ndarray,
        dl_den: Optional[np.ndarray],
        trg_pos: np.ndarray,
        setup: bool = True,
    ) -> np.ndarray:
        """Evaluate the potential at trg_pos due to sources at src_pos.

        With (kdim0, kdim1) = KERNEL_DIMS[kernel]: sl_den (single-layer) has
        kdim0 values per source, dl_den (double-layer density + normal) has
        kdim0+3 values per source, and the result has kdim1 values per
        target; either density may be None. Arrays are flat, in
        array-of-structures order. Pass setup=False when only densities (not
        positions) changed since the last call.
        """
        if src_pos.dtype != self.dtype:
            raise ValueError(
                f"Source array had the wrong dtype: {src_pos.dtype}. "
                f"This object was created with dtype {self.dtype}"
            )

        source_length = len(src_pos)
        if source_length % 3 != 0:
            raise ValueError(
                "Source arrays must have a length which is a multiple of 3"
            )
        n_src = source_length // 3
        kdim0, kdim1 = KERNEL_DIMS[self.kernel]

        if sl_den is not None:
            if sl_den.dtype != self.dtype:
                raise ValueError(
                    f"Source array had the wrong dtype: {sl_den.dtype}. "
                    f"This object was created with dtype {self.dtype}"
                )
            if len(sl_den) != n_src * kdim0:
                raise ValueError(
                    f"Single-layer density must have {kdim0} value(s) per "
                    f"source point for {self.kernel.name}"
                )
        if dl_den is not None:
            if dl_den.dtype != self.dtype:
                raise ValueError(
                    f"Source array had the wrong dtype: {dl_den.dtype}. "
                    f"This object was created with dtype {self.dtype}"
                )
            if len(dl_den) != n_src * (kdim0 + 3):
                raise ValueError(
                    f"Double-layer density must have {kdim0}+3 values per "
                    f"source point (density + normal) for {self.kernel.name}"
                )

        target_length = len(trg_pos)
        if target_length % 3 != 0:
            raise ValueError(
                "Target arrays must have a length which is a multiple of 3"
            )
        n_trg = target_length // 3
        trg_val = np.empty(n_trg * kdim1, dtype=self.dtype)

        get_function_dtype("PVFMMEval", self.dtype)(
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


class FMMVolumeTree:
    """Piecewise-Chebyshev volume discretization of a source density on an
    adaptive octree over [0,1]^3.

    Build with from_function() or from_coefficients(), then call evaluate()
    with a matching FMMVolumeContext.
    """

    def __init__(
        self,
        ptr: ctypes.c_void_p,
        cheb_deg: int,
        data_dim: int,
        n_trg: int,
        dtype: np.dtype,
    ):
        self._ptr = ptr
        self.cheb_deg = cheb_deg
        self.n_cheb = (cheb_deg + 1) ** 3
        self.n_coeff = (cheb_deg + 1) * (cheb_deg + 2) * (cheb_deg + 3) // 6
        self.data_dim = data_dim
        self.n_trg = n_trg
        self.dtype = dtype
        self._used_kernel = None

    @classmethod
    def from_function(
        cls,
        cheb_deg: int,
        data_dim: int,
        fn: Union[ffi.double_volume_callback, ffi.float_volume_callback],
        context: ctypes.c_void_p,
        trg_coord: np.ndarray,
        comm: MPI.Comm,
        tol: float,
        max_pts: int,
        periodic: Union[bool, FMMBoundaryType],
        init_depth: int,
    ) -> "FMMVolumeTree":
        """Build the tree by adaptively refining until the Chebyshev
        interpolation of fn (a C callback; see ffi.double_volume_callback)
        meets tol, with at most max_pts targets per leaf.
        """
        n_trg = len(trg_coord) // 3

        dtype = trg_coord.dtype
        ptr = get_function_dtype("PVFMMCreateVolumeTree", dtype)(
            cheb_deg,
            data_dim,
            fn,
            context,
            trg_coord,
            n_trg,
            ffi.get_MPI_COMM(comm),
            tol,
            max_pts,
            _boundary_value(periodic),
            init_depth,
        )
        return cls(ptr, cheb_deg, data_dim, n_trg, dtype)

    @classmethod
    def from_coefficients(
        cls,
        cheb_deg: int,
        data_dim: int,
        leaf_coord: np.ndarray,
        fn_coeff: np.ndarray,
        trg_coord: Optional[np.ndarray],
        comm: MPI.Comm,
        periodic: Union[bool, FMMBoundaryType],
    ) -> "FMMVolumeTree":
        """Build the tree from given leaf-node coordinates and Chebyshev
        coefficients of the source density (see nodes_to_coeff); trg_coord
        may be None.
        """
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
        dtype = leaf_coord.dtype
        if fn_coeff.dtype != dtype:
            raise ValueError(
                f"Mismatching dtypes. Leaves had dtype {dtype}, "
                f"but coefficients had dtype {fn_coeff.dtype}"
            )
        if trg_coord is not None:
            n_trg = len(trg_coord) // 3
            if trg_coord.dtype != dtype:
                raise ValueError(
                    f"Mismatching dtypes. Leaves had dtype {dtype}, "
                    f"but targets had dtype {trg_coord.dtype}"
                )
        else:
            n_trg = 0

        ptr = get_function_dtype("PVFMMCreateVolumeTreeFromCoeff", dtype)(
            N_leaf,
            cheb_deg,
            data_dim,
            leaf_coord,
            fn_coeff,
            trg_coord,
            n_trg,
            ffi.get_MPI_COMM(comm),
            _boundary_value(periodic),
        )
        return cls(ptr, cheb_deg, data_dim, n_trg, dtype)

    def __del__(self):
        if hasattr(self, "_ptr"):
            get_function_dtype("PVFMMDestroyVolumeTree", self.dtype)(
                ctypes.byref(ctypes.c_void_p(self._ptr))
            )

    def evaluate(self, fmm: FMMVolumeContext, loc_size: int) -> np.ndarray:
        """Run the volume FMM; returns the potential at the target points
        (n_trg * kernel-target-dimension values).
        """
        if fmm.dtype != self.dtype:
            raise ValueError(
                f"Volume context has dtype {fmm.dtype}, "
                f"but this tree has dtype {self.dtype}"
            )
        (_kdim0, kdim1) = KERNEL_DIMS[fmm.kernel]
        trg_val = np.empty(self.n_trg * kdim1, dtype=self.dtype)
        get_function_dtype("PVFMMEvaluateVolumeFMM", self.dtype)(
            trg_val, self._ptr, fmm._ptr, loc_size
        )
        self._used_kernel = fmm.kernel
        return trg_val

    def leaf_count(self) -> int:
        """Number of leaf nodes in the tree."""
        return int(get_function_dtype("PVFMMGetLeafCount", self.dtype)(self._ptr))

    def get_leaf_coordinates(self) -> np.ndarray:
        """Coordinates of the leaf-node corners (3 values per leaf)."""
        Nleaf = self.leaf_count()
        leaf_coord = np.empty(Nleaf * 3, dtype=self.dtype)
        get_function_dtype("PVFMMGetLeafCoord", self.dtype)(leaf_coord, self._ptr)
        return leaf_coord

    def get_coefficients(self) -> np.ndarray:
        """Chebyshev coefficients of the computed potential (requires a prior
        evaluate())."""
        if self._used_kernel is None:
            raise ValueError(
                "Cannot get coefficients of an un-evaluated tree"
            )  # TODO: is this true? what is the contract of this class
        n_leaf = self.leaf_count()
        (_kdim0, kdim1) = KERNEL_DIMS[self._used_kernel]
        coeff = np.empty(n_leaf * self.n_coeff * kdim1, dtype=self.dtype)
        get_function_dtype("PVFMMGetPotentialCoeff", self.dtype)(coeff, self._ptr)
        return coeff

    def get_values(self) -> np.ndarray:
        """Computed potential on the tensor-product Chebyshev nodes of each
        leaf (requires a prior evaluate())."""
        coeff = self.get_coefficients()
        n_leaf = self.leaf_count()
        (_kdim0, kdim1) = KERNEL_DIMS[self._used_kernel]
        value = np.empty(n_leaf * self.n_cheb * kdim1, dtype=self.dtype)
        get_function_dtype("PVFMMCoeff2Nodes", self.dtype)(
            value, n_leaf, self.cheb_deg, self.data_dim, coeff
        )
        return value
