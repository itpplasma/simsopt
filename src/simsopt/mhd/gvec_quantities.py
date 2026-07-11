# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np

from simsopt._core.optimizable import Optimizable
from simsopt._core.types import RealArray
from simsopt._core.util import Struct

if TYPE_CHECKING:
    from simsopt.mhd.gvec import Gvec


class GVECQuantity(Optimizable):
    """
    This optimizable computes a specified quantity as a dependent optimizable from the GVEC equilibrium.
    """

    def __init__(
        self,
        eq: Gvec,
        quantity: str,
        sfl: None | Literal["pest", "boozer"] = None,
        **kwargs,
    ):
        self.eq = eq
        self.quantity = quantity
        self.sfl = sfl
        self.kwargs = kwargs

        super().__init__(depends_on=[eq])

    @property
    def ev(self):
        """
        Evaluate the specified quantity from the GVEC state.

        Returns:
            The evaluated quantity as an xarray.Dataset.
        """
        if self.sfl is None:
            return self.eq.state.evaluate(self.quantity, **self.kwargs)
        else:
            return self.eq.state.evaluate_sfl(
                self.quantity, sfl=self.sfl, **self.kwargs
            )

    def J(self) -> np.ndarray:
        """Target function, returns a flattened array of the evaluated quantity."""
        return self.ev[self.quantity].data.flatten()

    def rms(self) -> float:
        """Root mean square of the evaluated quantity."""
        return np.sqrt(np.mean(self.ev[self.quantity] ** 2)).item()

    return_fn_map = {"J": J, "rms": rms}


class GvecQuasisymmetryRatioResidual(Optimizable):
    r"""Quasisymmetry ratio residual evaluated from a GVEC equilibrium.

    ``surfaces`` are normalized toroidal flux values. ``helicity_n`` follows
    the SIMSOPT/VMEC toroidal-angle convention; the GVEC-native toroidal mode
    has the opposite sign.
    """

    def __init__(
        self,
        eq: Gvec,
        surfaces: Union[float, RealArray],
        helicity_m: int = 1,
        helicity_n: int = 0,
        weights: Optional[RealArray] = None,
        ntheta: int = 63,
        nphi: int = 64,
    ) -> None:
        self.eq = eq
        self.surfaces = _surfaces(surfaces)
        self.helicity_m = _integer(helicity_m, "helicity_m")
        self.helicity_n = _integer(helicity_n, "helicity_n")
        if self.helicity_m == 0 and self.helicity_n == 0:
            raise ValueError("helicity_m and helicity_n must not both be zero")
        self.weights = _weights(weights, self.surfaces.size)
        self.ntheta = _positive_integer(ntheta, "ntheta")
        self.nphi = _positive_integer(nphi, "nphi")
        super().__init__(depends_on=[eq])

    def compute(self) -> Struct:
        state = self.eq.state
        nfp = _positive_integer(state.nfp, "state.nfp")
        rho = np.sqrt(self.surfaces)
        rho[self.surfaces == 0.0] = 1.0e-4
        theta = np.linspace(0.0, 2.0 * np.pi, self.ntheta, endpoint=False)
        zeta = np.linspace(
            0.0, 2.0 * np.pi / nfp, self.nphi, endpoint=False
        )
        ev = state.evaluate(
            "B",
            "mod_B",
            "grad_mod_B",
            "grad_rho",
            "dPhi_dr",
            "Jac",
            "iota",
            "B_theta_avg",
            "B_zeta_avg",
            rho=rho,
            theta=theta,
            zeta=zeta,
        )
        native_helicity_n = -self.helicity_n * nfp
        return _quasisymmetry_result(
            ev,
            self.weights,
            self.helicity_m,
            native_helicity_n,
            self.surfaces,
            rho,
        )

    def residuals(self) -> np.ndarray:
        return self.compute().residuals1d

    def profile(self) -> np.ndarray:
        return self.compute().profile

    def total(self) -> float:
        return self.compute().total


def _quasisymmetry_result(
    ev,
    weights: np.ndarray,
    helicity_m: int,
    native_helicity_n: int,
    surfaces: np.ndarray,
    rho: np.ndarray,
) -> Struct:
    import xarray as xr

    jacobian = ev.Jac.transpose("rad", "pol", "tor")
    mod_b = ev.mod_B.transpose("rad", "pol", "tor")
    if not np.all(np.isfinite(jacobian)) or np.any(jacobian <= 0.0):
        raise RuntimeError("GVEC quasisymmetry requires a positive finite Jacobian")
    if not np.all(np.isfinite(mod_b)) or np.any(mod_b <= 0.0):
        raise RuntimeError("GVEC quasisymmetry requires positive finite field strength")
    b_cross_grad_b_dot_grad_psi = xr.dot(
        xr.cross(ev.B, ev.grad_mod_B, dim="xyz"),
        ev.dPhi_dr * ev.grad_rho,
        dim="xyz",
    )
    b_dot_grad_b = xr.dot(ev.B, ev.grad_mod_B, dim="xyz")
    numerator = (
        (native_helicity_n - ev.iota * helicity_m) * b_cross_grad_b_dot_grad_psi
        - (helicity_m * ev.B_zeta_avg + native_helicity_n * ev.B_theta_avg)
        * b_dot_grad_b
    )
    normalizer = jacobian.sum(("pol", "tor"))
    if not np.all(np.isfinite(normalizer)) or np.any(normalizer <= 0.0):
        raise RuntimeError("GVEC quasisymmetry surface normalizer is not positive")
    surface_weights = xr.DataArray(
        weights, dims=("rad",), coords={"rad": jacobian.coords["rad"]}
    )
    weight = np.sqrt(surface_weights * jacobian / normalizer)
    residuals3d = (weight * numerator / mod_b**3).transpose("rad", "pol", "tor")
    residuals1d = np.asarray(residuals3d).reshape(-1)
    if not np.all(np.isfinite(residuals1d)):
        raise RuntimeError("GVEC quasisymmetry residuals are not finite")
    result = Struct()
    result.surfaces = surfaces.copy()
    result.rho = rho.copy()
    result.native_helicity_n = native_helicity_n
    result.residuals3d = np.asarray(residuals3d)
    result.residuals1d = residuals1d
    result.profile = np.sum(result.residuals3d**2, axis=(1, 2))
    result.total = float(result.residuals1d @ result.residuals1d)
    return result


def _surfaces(surfaces: Union[float, RealArray]) -> np.ndarray:
    values = np.atleast_1d(np.asarray(surfaces, dtype=float))
    if values.ndim != 1 or values.size == 0:
        raise ValueError("surfaces must be a nonempty scalar or one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("surfaces must be finite and lie in [0, 1]")
    return values.copy()


def _weights(weights: Optional[RealArray], count: int) -> np.ndarray:
    values = np.ones(count) if weights is None else np.asarray(weights, dtype=float)
    if values.shape != (count,):
        raise ValueError("weights must have one entry per surface")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    return values.copy()


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _positive_integer(value: int, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


class Elongation(Optimizable):
    """
    This optimizable computes the elongation of the plasma cross-sections using PCA.

    Applying a PCA on the boundary points gives us the principal axes for the cross-section.
    We can then take the minimum and maximum values along the principal axes to get an estimate for the "length" and "width" of the cross-section.

    With this definition, the values for elongation depend on the choice of cross-sections!
    """

    def __init__(self, eq: Gvec, zeta: Union[float, RealArray, Literal["int"]] = "int"):
        self.eq = eq
        self.zeta = zeta

        super().__init__(depends_on=[eq])

    def J(self):
        from scipy.linalg import svd

        ev = self.eq.state.evaluate(
            "pos", rho=[1.0], theta="int", zeta=self.zeta
        ).squeeze()

        elongation = np.zeros(len(ev.zeta))
        for z, zeta in enumerate(ev.zeta):
            pos = ev.pos.sel(zeta=zeta).squeeze().transpose("xyz", "pol").values
            # weighting with arclength
            tan = np.sqrt(
                np.sum(
                    (np.roll(pos, 1, axis=1) - np.roll(pos, -1, axis=1)) ** 2, axis=0
                )
            )
            tan /= np.sum(tan)
            cen0 = np.mean(pos, axis=1)
            cen = np.mean((pos - cen0[:, None]) * tan[None, :], axis=1) + cen0
            delta = pos - cen[:, None]

            # PCA
            sigma = delta * np.sqrt(tan[None, :])
            Sigma = sigma @ sigma.T
            U, S, V = svd(Sigma)

            u1 = U[:, 0]
            u2 = U[:, 1]
            # S[2] should be negligible (planar cross-sections)

            lim1 = np.array([np.min(delta.T @ u1), np.max(delta.T @ u1)])
            lim2 = np.array([np.min(delta.T @ u2), np.max(delta.T @ u2)])
            len1 = lim1[1] - lim1[0]
            len2 = lim2[1] - lim2[0]

            elongation[z] = len1 / len2
        return elongation

    def max(self):
        return np.max(self.J())

    return_fn_map = {"J": J, "max": max}
