# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Optional, Union
from collections.abc import Mapping

import numpy as np

try:
    import gvec
    from gvec import State
except ImportError:
    gvec = None
    State = None

from simsopt.geo import Surface, SurfaceScaled, SurfaceRZFourier
from simsopt.mhd.profiles import Profile, ProfilePolynomial, ProfileScaled
from simsopt.mhd.gvec_dofs import GVECSurfaceDoFs

logger = logging.getLogger("simsopt.mhd.gvec")


class GvecInterfaceMixin:
    def prepare_parameters(self, restart: Union[State, None] = None) -> Mapping:
        """
        Prepare DOFs as parameters for the GVEC input file as keyword arguments which can be passed to `run_stages`.
        """
        params = copy.deepcopy(self.parameters)

        # set paths to other files absolute
        for key in [
            "vmecwoutfile",
            "boundary_filename",
            "hmap_ncfile",
        ]:
            if key in params:
                params[key] = str(Path(params[key]).resolve())

        params = self.boundary_to_params(self.boundary, params)
        if not any(
            key in params for key in ["X1_a_cos", "X1_a_sin", "X2_a_cos", "X2_a_sin"]
        ):
            params["init_average_axis"] = True

        # non-RZphi coordinate frame
        if params.get("which_hmap", 1) != 1 and not (
            isinstance(self.boundary, GVECSurfaceDoFs)
            or (
                isinstance(self.boundary, SurfaceScaled)
                and isinstance(self.boundary.surf, GVECSurfaceDoFs)
            )
        ):
            raise TypeError(
                f"Non-RZphi coordinate frame (hmap={params.get('which_hmap')}) selected, but boundary is of type {type(self.boundary)}. "
                "Use GVECSurfaceDoFs to represent generic boundary DoFs."
            )

        # perturb boundary when restarting
        if restart:
            perturbation = gvec.util.CaseInsensitiveDict(boundary_perturb=True)
            for Xi in ["X1", "X2"]:
                for scs in ["sin", "cos"]:
                    perturbation[f"{Xi}pert_b_{scs}"] = {}
                    perturbation[f"{Xi}_b_{scs}"] = {}
                    restart_modes = restart.parameters.get(f"{Xi}_b_{scs}", {})
                    current_modes = params.get(f"{Xi}_b_{scs}", {})
                    # set boundary modes to values from restart
                    for (m, n), v in restart_modes.items():
                        perturbation[f"{Xi}_b_{scs}"][m, n] = v
                    # set boundary perturbation to difference between current and restart
                    for m, n in restart_modes.keys() | current_modes.keys():
                        delta = current_modes.get((m, n), 0) - restart_modes.get((m, n), 0)
                        if delta != 0.0:
                            perturbation[f"{Xi}pert_b_{scs}"][m, n] = delta
                    if len(perturbation[f"{Xi}pert_b_{scs}"]) == 0:
                        del perturbation[f"{Xi}pert_b_{scs}"]
                    if len(perturbation[f"{Xi}_b_{scs}"]) == 0:
                        del perturbation[f"{Xi}_b_{scs}"]
            params.update(perturbation)

        # profiles
        if self._iota is None and self._current is None:
            raise RuntimeError(
                "GVEC requires either an iota or a current profile to be set."
            )

        params["pres"] = self.profile_to_params(self._pressure)
        if self._iota is None:
            if "iota" in params:
                del params["iota"]
        else:
            params["iota"] = self.profile_to_params(self._iota)
        if self._current is None:  # iota constraint
            if "I_tor" in params:
                del params["I_tor"]
            if "picard_current" in params:
                del params["picard_current"]
        else:  # current optimization
            params["I_tor"] = self.profile_to_params(self._current)
            if "picard_current" not in params:
                params["picard_current"] = {} if "stages" in params else "auto"

        # local DOFs
        params["phiedge"] = self._phiedge
        return params

    @staticmethod
    def profile_to_params(profile: Profile) -> dict:
        """Convert a simsopt.Profile object to a dictionary of parameters for GVEC."""
        if isinstance(profile, ProfilePolynomial):
            return dict(type="polynomial", coefs=profile.local_full_x)
        elif isinstance(profile, ProfileScaled) and isinstance(
            profile.base, ProfilePolynomial
        ):
            return dict(
                type="polynomial",
                coefs=profile.base.local_full_x,
                scale=profile.get("scalefac"),
            )
        else:
            s = np.linspace(0, 1, 101)
            return dict(type="interpolation", vals=profile(s), rho2=s)

    @staticmethod
    def profile_from_params(params: Mapping) -> Profile:
        """Convert a dictionary of parameters to a simsopt.Profile object."""
        if params["type"] == "polynomial":
            profile = ProfilePolynomial(params["coefs"])
        elif params["type"] in ["bspline", "interpolation"]:
            raise NotImplementedError(
                f"Profile of type {params['type']} is not supported for converting a GVEC parameter file to a simsopt.Profile object. Use 'polynomial' instead."
            )
        else:
            raise ValueError(f"Unknown profile type {params['type']}")

        if "scale" in params:
            profile = ProfileScaled(profile, params["scale"])
        return profile

    @staticmethod
    def boundary_to_params(
        boundary: Union[Surface, SurfaceScaled, GVECSurfaceDoFs],
        append: Optional[Mapping] = None,
    ) -> dict:
        """Convert a simsopt.SurfaceRZFourier object into GVEC boundary parameters.

        The output parameters will include the (non-boundary) contents of the `append` dictionary, if provided.
        """
        if isinstance(boundary, SurfaceScaled):
            boundary = boundary.surf
        if not isinstance(boundary, (SurfaceRZFourier, GVECSurfaceDoFs)):
            boundary = boundary.to_RZFourier()

        params = boundary.to_gvec_parameters()

        if append is not None:
            params = copy.deepcopy(append) | params

        return params

    @staticmethod
    def boundary_from_params(
        params: Mapping,
    ) -> Union[SurfaceRZFourier, GVECSurfaceDoFs]:
        """Convert a dictionary of parameters to a simsopt.SurfaceRZFourier object.

        Note that simsopt assumes a (right-handed) (R,phi,Z) coordinate system,
        while GVEC uses a (R,Z,phi) coordinate system. The toroidal angle therefore increases
        in the clockwise, rather than counter-clockwise direction, when viewed from above.
        """

        # RZphi with phi clockwise when viewed from above
        if params.get("which_hmap", 1) == 1:
            boundary = SurfaceRZFourier.from_gvec_parameters(params)

        # generic boundary type: only contains boundary modes, cannot be evaluated
        else:
            boundary = GVECSurfaceDoFs.from_gvec_parameters(params)

        boundary.fix_all()
        return boundary

    @property
    def state(self) -> State:
        """Return the gvec.State object, representing the equilibrium, rerunning GVEC if necessary.

        The State object allows evaluating the configuration at any position
        using the same accuracy and discretization as used during the minimization.

        As this is not a number, this is not a 'return function' in the usual sense.

        Indirectly raises an ObjectiveFailure if GVEC fails to run.
        """
        self.run()
        return self._state

    # === INPUT VARIABLES === #

    @property
    def phiedge(self) -> float:
        """The toroidal magnetic flux at the last closed flux surface."""
        return self._phiedge

    @phiedge.setter
    def phiedge(self, phiedge: float):
        """Set the toroidal magnetic flux at the last closed flux surface."""
        if phiedge == self._phiedge:
            return
        logging.debug(f"setting phiedge to {phiedge}")
        self._phiedge = phiedge
        self.set_recompute_flag()

    @property
    def boundary(self) -> Surface:
        """The plasma boundary as a simsopt.Surface object.

        Note that the default coordinate system for VMEC uses a toroidal angle in the opposite
        direction of GVEC's toroidal angle (counter-clockwise vs clockwise when viewed from above).
        """
        return self._boundary

    @boundary.setter
    def boundary(self, boundary: Surface):
        """Set the plasma boundary."""
        if boundary is self._boundary:
            return
        logging.debug("setting plasma boundary")
        self.remove_parent(self._boundary)
        self._boundary = boundary
        self.append_parent(self._boundary)
        # self.set_recompute_flag()  # also called by append_parent()

    @property
    def pressure_profile(self) -> Profile:
        """The pressure profile in terms of the normalized toroidal flux $s$."""
        return self._pressure

    @pressure_profile.setter
    def pressure_profile(self, pressure_profile: Union[Profile, float]):
        if pressure_profile is self._pressure:
            return
        logging.debug("setting pressure profile")
        if isinstance(pressure_profile, float):
            pressure_profile = ProfilePolynomial([pressure_profile])
        self.remove_parent(self._pressure)
        self._pressure = pressure_profile
        self.append_parent(self._pressure)
        # self.set_recompute_flag()  # called by append_parent()

    @property
    def iota_profile(self) -> Union[Profile, None]:
        """
        The rotational transform profile in terms of the normalized toroidal flux $s$.

        To evaluate the rotational transform from the equilibrium, use `iota`.
        If `current_profile` is set, this profile is only used to set the initial iota profile.

        Note that the default coordinate system for VMEC uses a toroidal angle in the opposite
        direction of GVEC's toroidal angle (counter-clockwise vs clockwise when viewed from above).
        Therefore the iota profile typically has the opposite sign compared to VMEC.
        """
        return self._iota

    @iota_profile.setter
    def iota_profile(self, iota_profile: Union[Profile, float, None]):
        if iota_profile is self._iota:
            return
        logger.debug("setting iota profile")
        if isinstance(iota_profile, float):
            iota_profile = ProfilePolynomial([iota_profile])
        if self._current is None and self._iota is not None:
            self.remove_parent(self._iota)
        self._iota = iota_profile
        if self._current is None and iota_profile is not None:
            self.append_parent(self._iota)
        # self.set_recompute_flag()  # called by append_parent()

    @property
    def current_profile(self) -> Union[Profile, None]:
        """
        The toroidal current profile in terms of the normalized toridal flux $s$.

        To evaluate the toroidal current from the equilibrium, use `I_tor`.
        If this profile is set, GVEC will run using "current-optimization",
        otherwise it will run with "iota-constraint".

        Note that the default coordinate system for VMEC uses a toroidal angle in the opposite
        direction of GVEC's toroidal angle (counter-clockwise vs clockwise when viewed from above).
        Therefore the current profile typically has the opposite sign compared to VMEC.
        """
        return self._current

    @current_profile.setter
    def current_profile(self, current_profile: Union[Profile, float, None]):
        if current_profile is self._current:
            return
        logging.debug("setting toroidal current profile")
        if isinstance(current_profile, float):
            current_profile = ProfilePolynomial([current_profile])
        if self._current is None and self._iota is not None:
            self.remove_parent(self._iota)
        elif self._current is not None:
            self.remove_parent(self._current)
        self._current = current_profile
        if current_profile is None and self._iota is not None:
            self.append_parent(self._iota)
        elif current_profile is not None:
            self.append_parent(self._current)
        # self.set_recompute_flag()  # called by append_parent()

    # === RETURN FUNCTIONS - SIMILAR TO VMEC === #
    # These functions provide some common optimization targets,
    # compatible with the VMEC optimizable. It is recommended to
    # use the more general 'GVECQuantity' optimizable instead.
    # As 'return_functions' they are not 'Optimizable' objects.
    # They all use 'self.state' which checks whether a run is required.
    # Some deviations from the values returned by VMEC are expected,
    # due to numerical differences and differently defined coordinate systems.

    def aspect(self):
        """Return the effective aspect ratio."""
        ev = self.state.evaluate("aspect_ratio")
        return ev.aspect_ratio.item()

    def volume(self):
        """Return the volume inside the last closed flux surface."""
        ev = self.state.evaluate("V")
        return ev.V.item()

    def iota_axis(self):
        """Return the rotational transform on axis."""
        ev = self.state.evaluate("iota", rho=1e-4)
        return ev.iota.item()

    def iota_edge(self):
        """Return the rotational transform at the boundary."""
        ev = self.state.evaluate("iota", rho=1.0)
        return ev.iota.item()

    def mean_iota(self):
        """Return the mean rotational transform.

        The average is taken with respect to the toroidal flux, i.e. $\\rho^2$.
        """
        ev = self.state.evaluate("iota_avg2")
        return ev.iota_avg2.item()

    def mean_shear(self):
        """Return the average magnetic shear.

        The average is taken with respect to the toroidal flux, i.e. $\\rho^2$.
        """
        ev = self.state.evaluate("shear_avg2")
        return ev.shear_avg2.item()

    def vacuum_well(self) -> float:
        """Return the depth of the vacuum magnetic well."""
        ev = self.state.evaluate("vacuum_magnetic_well_depth")
        return ev.vacuum_magnetic_well_depth.item()

    return_fn_map = {
        "aspect": aspect,
        "volume": volume,
        "iota_axis": iota_axis,
        "iota_edge": iota_edge,
        "mean_iota": mean_iota,
        "mean_shear": mean_shear,
        "vacuum_well": vacuum_well,
    }
