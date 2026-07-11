# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
This module provides a class that provides an optimizable for the GVEC equilibrium code.

The Gvec optimizable class is designed to be similar to the Vmec optimizable, but the interfaces are not identical.
"""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path
from typing import Optional, Literal, Union
from collections.abc import Mapping
import shutil
import tempfile

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
    from simsopt.util.mpi import MpiPartition
except ImportError as e:
    MPI = None
    MpiPartition = None
    logger.debug(str(e))

try:
    import gvec
    from gvec import State
    from gvec.errors import GVECError
except ImportError as e:
    gvec = None
    State = None
    GVECError = RuntimeError
    logger.debug(str(e))

from simsopt._core.optimizable import Optimizable
from simsopt._core.util import ObjectiveFailure
from simsopt.geo import Surface, SurfaceRZFourier
from simsopt.mhd.profiles import Profile, ProfilePolynomial
from simsopt.mhd.gvec_dofs import GVECSurfaceDoFs
from simsopt.mhd.gvec_interface import GvecInterfaceMixin
from simsopt.mhd.gvec_quantities import Elongation as Elongation
from simsopt.mhd.gvec_quantities import GVECQuantity as GVECQuantity
from simsopt.mhd.gvec_quantities import (
    GvecQuasisymmetryRatioResidual as GvecQuasisymmetryRatioResidual,
)

__all__ = [
    "Gvec",
    "GVECQuantity",
    "GVECSurfaceDoFs",
    "GvecQuasisymmetryRatioResidual",
]

default_parameters = dict(
    ProjectName="SIMSOPT-GVEC",
    X1X2_deg=5,
    LA_deg=5,
    totalIter=10000,
    minimize_tol=1e-5,
    sgrid=dict(
        grid_type=4,
        nElems=5,
    ),
)


def _parameter_signature(value):
    if isinstance(value, Mapping):
        return tuple(
            sorted((str(key).casefold(), _parameter_signature(item)) for key, item in value.items())
        )
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, (list, tuple)):
        return tuple(_parameter_signature(item) for item in value)
    if isinstance(value, float) and math.isnan(value):
        return ("nan",)
    return value


class Gvec(GvecInterfaceMixin, Optimizable):
    """
    This class represents the optimizable for the GVEC equilibrium code.

    Args:
        phiedge: the prescribed total toroidal magnetic flux.
        boundary: the prescribed plasma boundary.
        pressure: the prescribed pressure profile.
        iota: the prescribed (or initial) rotational transform profile.
        current: the prescribed toroidal current profile. if None, GVEC will run in prescribed-iota mode, otherwise it will run in prescribed-current mode with the given iota as initial guess.
        parameters: a dictionary of GVEC parameters.
        restart: ...
        delete_intermediates: whether to delete intermediate files after the run.
        save_diagnostics: whether to save the minimization diagnostic plot for every run.
        require_convergence: whether force and prescribed-current tolerances must be met.
        mpi: the MPI partition to use. For now, GVEC is best run with ngroups = nprocs and multiple OpenMP threads assigned to each MPI process.
    """

    def __init__(
        self,
        phiedge: Optional[float] = None,
        boundary: Optional[Surface] = None,
        pressure: Optional[Profile] = None,
        iota: Union[Profile, float, None] = None,
        current: Union[Profile, float, None] = None,
        parameters: Optional[Mapping] = None,
        restart: Union[Literal["first", "last"], Path, str, None] = None,
        delete_intermediates: bool = True,
        keep_failures: bool = False,
        keep_gvec_intermediates: Optional[Literal["all", "stages"]] = None,
        save_diagnostics: bool = False,
        require_convergence: bool = True,
        mpi: Optional[MpiPartition] = None,
    ):
        if gvec is None:
            raise RuntimeError(
                "The Gvec Optimizable requires the 'gvec' package to be installed!"
            )

        # init arguments which are not DoFs
        self.parameters = gvec.util.CaseInsensitiveDict(copy.deepcopy(parameters or {}))
        self.delete_intermediates = delete_intermediates
        self.keep_failures = keep_failures
        self.keep_gvec_intermediates = keep_gvec_intermediates
        self.save_diagnostics = save_diagnostics
        self.require_convergence = require_convergence
        self.mpi = mpi

        if restart in [None, "first", "last"]:
            self.restart = restart
        elif isinstance(restart, (str, Path)):
            if not Path(restart).is_dir():
                raise ValueError(
                    f"restart path {restart} does not exist or is not a directory."
                )
            self.restart = gvec.find_state(restart)
        else:
            raise ValueError(
                "restart must be 'first', 'last', a path to a GVEC run directory or None."
            )

        # auxiliary attributes
        # number of times GVEC has been run (0-indexed, this MPI-process only)
        self.run_count: int = -1
        # flag for caching e.g. set after changing parameters
        self.run_required: bool = True
        # flag for whether the last run was successful
        self.run_successful: bool = False
        # state object representing the GVEC equilibrium
        self._state: State | None = None
        # the gvec run object, used to access the state & diagnostics
        self._runobj: gvec.Run = None
        self.rundir: Path | None = None
        self._rundir_deletion_list: list = []
        self._first_state: State | None = None
        self._last_parameter_signature = None

        # init MPI
        if MPI:
            if self.mpi is None:
                self.mpi = MpiPartition()  # default to ngroups = nprocs
            self._mpi_id = f"_{self.mpi.rank_world:03d}"
        else:
            if self.mpi:
                logger.warning("MPI not available, ignoring MpiPartition")
            self._mpi_id = ""

        # fill default parameters
        for key, value in default_parameters.items():
            if key not in self.parameters:
                self.parameters[key] = copy.deepcopy(value)
            elif isinstance(value, Mapping):
                for subkey, subvalue in value.items():
                    if subkey not in self.parameters[key]:
                        self.parameters[key][subkey] = subvalue

        # init DoFs
        if phiedge is None:
            if "phiedge" in self.parameters:
                phiedge = self.parameters["phiedge"]
            else:
                phiedge = 1.0
        self._phiedge = phiedge

        if boundary is None:
            if "X1_mn_max" in self.parameters:
                boundary = self.boundary_from_params(self.parameters)
            else:
                boundary = SurfaceRZFourier()
        self._boundary = boundary

        if pressure is None:
            if "pres" in self.parameters:
                pressure = self.profile_from_params(self.parameters["pres"])
            else:
                pressure = ProfilePolynomial([0.0])
        elif isinstance(pressure, float):
            pressure = ProfilePolynomial([pressure])
        self._pressure = pressure

        if iota is None and "iota" in self.parameters:
            iota = self.profile_from_params(self.parameters["iota"])
        elif isinstance(iota, float):
            iota = ProfilePolynomial([iota])
        self._iota = iota

        if current is None and "I_tor" in self.parameters:
            current = self.profile_from_params(self.parameters["I_tor"])
        elif isinstance(current, float):
            current = ProfilePolynomial([current])
        elif current is None and iota is None:
            current = ProfilePolynomial([0.0])
        self._current = current

        # init Optimizable
        x0 = self.get_dofs()
        fixed = np.full(len(x0), True)
        names = ["phiedge"]
        depends = [self._boundary, self._pressure]
        if current is not None:
            depends += [self._current]
        elif iota is not None:
            depends += [self._iota]
        super().__init__(
            x0=x0,
            fixed=fixed,
            names=names,
            depends_on=depends,
            external_dof_setter=self.__class__.set_dofs,
        )

    def __copy__(self):
        restart = self.restart if self.restart is None or isinstance(self.restart, str) else None
        duplicate = self.__class__(
            phiedge=self.phiedge,
            boundary=self.boundary,
            pressure=self.pressure_profile,
            iota=self.iota_profile,
            current=self.current_profile,
            parameters=copy.deepcopy(self.parameters),
            restart=restart,
            delete_intermediates=self.delete_intermediates,
            keep_failures=self.keep_failures,
            keep_gvec_intermediates=self.keep_gvec_intermediates,
            save_diagnostics=self.save_diagnostics,
            require_convergence=self.require_convergence,
            mpi=self.mpi,
        )
        if restart is None and self.restart is not None:
            duplicate.restart = self.restart
        for index, (name, is_free) in enumerate(
            zip(self.local_full_dof_names, self.local_dofs_free_status)
        ):
            duplicate.set_lower_bound(name, self.local_full_lower_bounds[index])
            duplicate.set_upper_bound(name, self.local_full_upper_bounds[index])
            if is_free:
                duplicate.unfix(name)
        return duplicate

    @classmethod
    def from_parameter_file(
        cls,
        parameter_file: Union[str, Path],
        **kwargs,
    ):
        parameters = gvec.util.read_parameters(parameter_file)
        return cls(parameters=parameters, **kwargs)

    @classmethod
    def from_rundir(
        cls,
        rundir: Union[str, Path],
        **kwargs,
    ):
        physical_overrides = {
            "phiedge",
            "boundary",
            "pressure",
            "iota",
            "current",
            "parameters",
        }.intersection(kwargs)
        if physical_overrides:
            names = ", ".join(sorted(physical_overrides))
            raise ValueError(f"from_rundir does not accept equilibrium overrides: {names}")
        state = gvec.find_state(rundir)
        parameter_toml = list(Path(rundir).glob("parameter*.toml"))
        if len(parameter_toml) == 1:
            self = cls.from_parameter_file(parameter_toml[0], **kwargs)
        else:
            self = cls.from_parameter_file(state.parameterfile, **kwargs)
        self._state = state
        self.run_required = False
        self.run_successful = True
        self.rundir = rundir
        self._last_parameter_signature = _parameter_signature(self.parameters)
        return self

    def recompute_bell(self, parent=None):
        """Set the recomputation flag"""
        logger.debug(
            f"recompute_bell called by {parent}: run_required = {self.run_required}"
        )
        self.run_required = True

    def get_dofs(self):
        """Return the DOFs owned by the GVEC optimizable.
        o
         This does not include any DoFs owned by dependent objects (boundary or profiles).
        """
        return np.array([self._phiedge])

    def set_dofs(self, x):
        """Set the DoFs owned by the GVEC optimizable.

        This does not include any DoFs owned by dependent objects (boundary or profiles).
        """
        if len(x) != 1:
            raise ValueError(f"Expected 1 DOF, got {len(x)}")
        self.phiedge = x[0]

    @property
    def logger(self):
        """The gvec internal logger, separate from the simsopt logger."""
        return logging.getLogger("gvec")

    def run(self, force: bool = False) -> None:
        """
        Run GVEC (via subprocess) if `run_required` or `force` is True.
        """
        parameter_signature = _parameter_signature(self.parameters)
        if parameter_signature != self._last_parameter_signature:
            self.run_required = True

        if not self.run_required:
            if force:
                logger.debug("re-run forced")
            elif not self.run_successful:
                logger.debug("no run required, cached run not successful")
                raise ObjectiveFailure("cached GVEC run was not successful")
            else:
                logger.debug("no run required")
                return

        self.run_count += 1
        logger.debug(f"preparing to run GVEC run number {self.run_count}")

        # configure restart
        if self.restart is None:
            restart = None
            logger.info("running GVEC")
        elif isinstance(self.restart, State):
            restart = self.restart
            logger.info(f"running GVEC from {self.restart}")
        elif self.restart == "first":
            if self._first_state is None:
                restart = None
                logger.info("running GVEC (first run)")
            else:
                restart = self._first_state
                logger.info("running GVEC from first run")
        elif self.restart == "last":
            if self.run_successful:
                restart = self._state
                logger.info("running GVEC from previous run")
            else:
                restart = None
                logger.info("running GVEC (previous run not successful)")
        else:
            raise RuntimeError(f"invalid value for restart: {self.restart}")

        # prepare parameter file
        params = self.prepare_parameters(restart)

        self.rundir = Path(
            tempfile.mkdtemp(
                prefix=f"gvec{self._mpi_id}-{self.run_count:03d}-",
                dir=".",
            )
        )
        try:
            gvec.util.write_parameters(params, self.rundir / "parameters.toml")
        except Exception:
            if not self.keep_failures:
                shutil.rmtree(self.rundir, ignore_errors=True)
            raise

        # run GVEC
        self.run_required = False
        self.run_successful = False
        self._state = None
        self._last_parameter_signature = parameter_signature

        try:
            self._runobj = gvec.run(
                params,
                restart,
                runpath=self.rundir,
                quiet=True,
                keep_intermediates=self.keep_gvec_intermediates,
            )
            candidate_state = self._runobj.state
            if self.require_convergence:
                convergence_error = self._convergence_error()
                if convergence_error:
                    raise RuntimeError(convergence_error)
        except Exception as e:
            logger.error(f"GVEC failed with: {e}")
            if not self.keep_failures:
                shutil.rmtree(self.rundir, ignore_errors=True)
            if isinstance(e, (RuntimeError, GVECError)):
                raise ObjectiveFailure("Run GVEC failed.") from e
            self.run_required = True
            raise

        self._state = candidate_state
        self.run_successful = True
        if self.restart == "first" and self._first_state is None:
            self._first_state = self._state
        if self.delete_intermediates and self._state is not self._first_state:
            self._rundir_deletion_list.append(self.rundir)
        logger.debug(f"GVEC finished in {self._runobj.GVEC_iter_used} iterations")

        if self.save_diagnostics:
            self._save_diagnostics()

        # remove old rundirs
        while self._rundir_deletion_list and self._rundir_deletion_list[0] != self.rundir:
            rundir = self._rundir_deletion_list.pop(0)
            logger.debug(f"deleting {rundir}")
            shutil.rmtree(rundir)

    def _save_diagnostics(self):
        fig = None
        try:
            fig = self._runobj.plot_diagnostics_minimization()
            fig.savefig(self.rundir / "iterations.png")
        except Exception as error:
            logger.warning(f"Could not save GVEC diagnostics: {error}")
        finally:
            if fig is not None:
                try:
                    plt.close(fig)
                except Exception as error:
                    logger.warning(f"Could not close GVEC diagnostics: {error}")

    def _convergence_error(self):
        state_parameters = self._runobj._state_parameters
        force = float(self._runobj.max_force)
        force_tolerance = float(state_parameters["minimize_tol"])
        failures = []
        if not np.isfinite(force) or force > force_tolerance:
            failures.append(f"force {force:.6e} exceeds tolerance {force_tolerance:.6e}")
        if self._runobj.curr_constraint:
            iota_error = float(self._runobj.rms_iota)
            iota_tolerance = self._current_iota_tolerance(state_parameters)
            if iota_tolerance is None:
                failures.append("prescribed-current run reported no iota tolerance")
            elif not np.isfinite(iota_error) or iota_error > iota_tolerance:
                failures.append(
                    f"current-profile iota error {iota_error:.6e} exceeds tolerance "
                    f"{iota_tolerance:.6e}"
                )
        return "; ".join(failures)

    def _current_iota_tolerance(self, state_parameters):
        picard_current = state_parameters.get("picard_current")
        if isinstance(picard_current, Mapping) and "iota_tol" in picard_current:
            return float(picard_current["iota_tol"])
        executed_stages = zip(self._runobj.stages, self._runobj.n_runs_in_stage)
        for stage, run_count in reversed(list(executed_stages)):
            if run_count == 0:
                continue
            picard_current = stage.get("picard_current")
            if isinstance(picard_current, Mapping) and "iota_tol" in picard_current:
                return float(picard_current["iota_tol"])
        return None
