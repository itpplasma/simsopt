# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union

import numpy as np

from simsopt._core.optimizable import Optimizable
from simsopt._core.types import RealArray

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
