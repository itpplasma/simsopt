import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from matplotlib.figure import Figure
from monty.tempfile import ScratchDir

from simsopt._core.util import ObjectiveFailure
from simsopt.geo import SurfaceRZFourier
from simsopt.mhd.gvec import Gvec

try:
    import gvec
except ImportError:
    gvec = None


class _FakeFigure(Figure):
    def savefig(self, path):
        Path(path).touch()


class _FakeRun:
    def __init__(self, state):
        self.state = state
        self.GVEC_iter_used = 1
        self.diagnostics_calls = 0
        self.max_force = 1.0e-6
        self.curr_constraint = False
        self._state_parameters = {"minimize_tol": 1.0e-5}
        self.stages = []
        self.n_runs_in_stage = []

    def plot_diagnostics_minimization(self):
        self.diagnostics_calls += 1
        return _FakeFigure()


class _FakeState:
    def __init__(self):
        self.parameters = {
            "X1_b_sin": {},
            "X1_b_cos": {},
            "X2_b_sin": {},
            "X2_b_cos": {},
        }


class _FailingDiagnosticsRun(_FakeRun):
    def plot_diagnostics_minimization(self):
        raise OSError("diagnostics unavailable")


class _StateFailureRun:
    def __init__(self):
        self.GVEC_iter_used = 1
        self.max_force = 1.0e-6
        self.curr_constraint = False
        self._state_parameters = {"minimize_tol": 1.0e-5}

    @property
    def state(self):
        raise ValueError("state unavailable")


@unittest.skipIf(gvec is None, "gvec package not installed")
class GvecLifecycleTests(unittest.TestCase):
    def test_default_parameters_are_not_shared(self):
        first = Gvec()
        second = Gvec()
        first.parameters["sgrid"]["nElems"] = 99
        self.assertEqual(second.parameters["sgrid"]["nElems"], 5)

    def test_nested_parameter_change_invalidates_cached_state(self):
        first_state = object()
        second_state = object()
        eq = Gvec(delete_intermediates=False)
        with ScratchDir("/tmp"), patch(
            "simsopt.mhd.gvec.gvec.run",
            side_effect=[_FakeRun(first_state), _FakeRun(second_state)],
        ) as run, patch("simsopt.mhd.gvec.gvec.util.write_parameters"):
            eq.run()
            eq.parameters["sgrid"]["nElems"] = 9
            eq.run()
        self.assertEqual(run.call_count, 2)
        self.assertIs(eq.state, second_state)

    def test_instances_use_disjoint_run_directories(self):
        first = Gvec()
        first.unfix("phiedge")
        first.set_lower_bound("phiedge", -2.0)
        first.set_upper_bound("phiedge", 2.0)
        second = copy.copy(first)
        self.assertIsNot(first._dofs, second._dofs)
        self.assertIsNot(first.parents, second.parents)
        self.assertIsNot(first._children, second._children)
        self.assertNotEqual(first.name, second.name)
        self.assertNotEqual(hash(first), hash(second))
        self.assertTrue(first.is_free("phiedge"))
        self.assertTrue(second.is_free("phiedge"))
        self.assertEqual(second.local_full_lower_bounds[0], -2.0)
        self.assertEqual(second.local_full_upper_bounds[0], 2.0)
        second.fix("phiedge")
        second.phiedge = 2.0
        self.assertTrue(first.is_free("phiedge"))
        self.assertEqual(first.phiedge, 1.0)
        self.assertTrue(any(child() is second for child in first.boundary._children))
        with ScratchDir("/tmp"), patch(
            "simsopt.mhd.gvec.gvec.run",
            side_effect=[_FakeRun(object()) for _ in range(4)],
        ), patch("simsopt.mhd.gvec.gvec.util.write_parameters"):
            first.run()
            second.run()
            second_directory = second.rundir
            first.run(force=True)
            first_directory = first.rundir
            self.assertTrue(second_directory.exists())
            second.run(force=True)
            self.assertTrue(first_directory.exists())
        self.assertIsNot(first._rundir_deletion_list, second._rundir_deletion_list)

    def test_diagnostics_are_opt_in(self):
        without_diagnostics = _FakeRun(object())
        with_diagnostics = _FakeRun(object())
        first = Gvec(delete_intermediates=False)
        second = Gvec(delete_intermediates=False, save_diagnostics=True)
        with ScratchDir("/tmp"), patch(
            "simsopt.mhd.gvec.gvec.run",
            side_effect=[without_diagnostics, with_diagnostics],
        ), patch("simsopt.mhd.gvec.gvec.util.write_parameters"):
            first.run()
            second.run()
        self.assertEqual(without_diagnostics.diagnostics_calls, 0)
        self.assertEqual(with_diagnostics.diagnostics_calls, 1)

    def test_diagnostics_failure_does_not_change_equilibrium_success(self):
        state = object()
        eq = Gvec(delete_intermediates=False, save_diagnostics=True)
        with ScratchDir("/tmp"), patch(
            "simsopt.mhd.gvec.gvec.run",
            return_value=_FailingDiagnosticsRun(state),
        ), patch("simsopt.mhd.gvec.gvec.util.write_parameters"):
            eq.run()
        self.assertTrue(eq.run_successful)
        self.assertIs(eq.state, state)

    def test_first_restart_survives_repeated_cleanup(self):
        states = [_FakeState(), object(), object()]
        eq = Gvec(restart="first", delete_intermediates=True)
        with ScratchDir("/tmp"), patch(
            "simsopt.mhd.gvec.gvec.run",
            side_effect=[_FakeRun(state) for state in states],
        ) as run, patch("simsopt.mhd.gvec.gvec.util.write_parameters"):
            eq.run()
            first_directory = eq.rundir
            eq.run(force=True)
            eq.run(force=True)
            self.assertTrue(first_directory.exists())
        self.assertIsNone(run.call_args_list[0].args[1])
        self.assertIs(run.call_args_list[1].args[1], states[0])
        self.assertIs(run.call_args_list[2].args[1], states[0])

    def test_failed_first_run_can_recover_without_restart(self):
        eq = Gvec(restart="first", delete_intermediates=False)
        error = gvec.errors.InvalidParameterError("invalid")
        recovered_state = object()
        with ScratchDir("/tmp"), patch(
            "simsopt.mhd.gvec.gvec.run",
            side_effect=[error, _FakeRun(recovered_state)],
        ) as run, patch("simsopt.mhd.gvec.gvec.util.write_parameters"):
            with self.assertRaises(ObjectiveFailure):
                eq.run()
            eq.run(force=True)
        self.assertIsNone(run.call_args_list[0].args[1])
        self.assertIsNone(run.call_args_list[1].args[1])

    def test_gvec_failure_is_translated_and_removed(self):
        eq = Gvec(keep_failures=False)
        error = gvec.errors.InvalidParameterError("invalid")
        with ScratchDir("/tmp") as directory, patch(
            "simsopt.mhd.gvec.gvec.run", side_effect=error
        ) as run, patch("simsopt.mhd.gvec.gvec.util.write_parameters"):
            with self.assertRaises(ObjectiveFailure):
                eq.run()
            failed_directory = Path(directory) / eq.rundir
            with self.assertRaises(ObjectiveFailure):
                eq.run()
            self.assertFalse(failed_directory.exists())
        self.assertEqual(run.call_count, 1)

    def test_nonconverged_equilibrium_is_an_objective_failure(self):
        result = _FakeRun(object())
        result.max_force = 2.0e-5
        eq = Gvec(keep_failures=False)
        with ScratchDir("/tmp"), patch(
            "simsopt.mhd.gvec.gvec.run", return_value=result
        ), patch("simsopt.mhd.gvec.gvec.util.write_parameters"):
            with self.assertRaisesRegex(ObjectiveFailure, "Run GVEC failed"):
                eq.run()
            self.assertFalse(eq.rundir.exists())
        self.assertFalse(eq.run_successful)

    def test_nonconverged_current_profile_is_an_objective_failure(self):
        result = _FakeRun(object())
        result.curr_constraint = True
        result.rms_iota = 2.0e-5
        result._state_parameters["picard_current"] = {"iota_tol": 1.0e-5}
        eq = Gvec(keep_failures=False)
        with ScratchDir("/tmp"), patch(
            "simsopt.mhd.gvec.gvec.run", return_value=result
        ) as run, patch("simsopt.mhd.gvec.gvec.util.write_parameters"):
            with self.assertRaisesRegex(ObjectiveFailure, "Run GVEC failed"):
                eq.run()
            with self.assertRaisesRegex(ObjectiveFailure, "cached GVEC run"):
                eq.run()
            self.assertFalse(eq.rundir.exists())
        self.assertEqual(run.call_count, 1)
        self.assertFalse(eq.run_successful)

    def test_final_picard_off_still_requires_current_convergence(self):
        result = _FakeRun(object())
        result.curr_constraint = True
        result.rms_iota = 1.0
        result._state_parameters["picard_current"] = "off"
        result.stages = [
            {"picard_current": {"iota_tol": 1.0e-5, "target": "iota_and_force"}},
            {"picard_current": "off"},
        ]
        result.n_runs_in_stage = [1, 1]
        eq = Gvec(keep_failures=False)
        with ScratchDir("/tmp"), patch(
            "simsopt.mhd.gvec.gvec.run", return_value=result
        ), patch("simsopt.mhd.gvec.gvec.util.write_parameters"):
            with self.assertRaisesRegex(ObjectiveFailure, "Run GVEC failed"):
                eq.run()
            self.assertFalse(eq.rundir.exists())
        self.assertFalse(eq.run_successful)

    def test_unexecuted_current_stage_does_not_set_convergence_tolerance(self):
        state = object()
        result = _FakeRun(state)
        result.curr_constraint = True
        result.rms_iota = 5.0e-4
        result._state_parameters["picard_current"] = "off"
        result.stages = [
            {"picard_current": {"iota_tol": 1.0e-3, "target": "iota_and_force"}},
            {"picard_current": {"iota_tol": 1.0e-8, "target": "iota_and_force"}},
            {"picard_current": "off"},
        ]
        result.n_runs_in_stage = [1, 0, 0]
        eq = Gvec(delete_intermediates=False)
        with ScratchDir("/tmp"), patch(
            "simsopt.mhd.gvec.gvec.run", return_value=result
        ), patch("simsopt.mhd.gvec.gvec.util.write_parameters"):
            eq.run()

        self.assertTrue(eq.run_successful)
        self.assertIs(eq.state, state)

    def test_restart_perturbation_can_zero_or_remove_boundary_modes(self):
        surface = SurfaceRZFourier(nfp=1, mpol=1, ntor=1, stellsym=True)
        surface.set_rc(0, 0, 5.0)
        surface.set_rc(1, 0, 1.0)
        surface.set_zs(1, 0, 1.0)
        restart = _FakeState()
        restart.parameters["X1_b_cos"] = {(1, 1): 0.2, (2, 2): -0.3}
        for restart_kind in ("first", "last"):
            eq = Gvec(boundary=surface, iota=0.05, restart=restart_kind)

            parameters = eq.prepare_parameters(restart)

            self.assertEqual(parameters["X1_b_cos"][(1, 1)], 0.2)
            self.assertEqual(parameters["X1_b_cos"][(2, 2)], -0.3)
            self.assertEqual(parameters["X1pert_b_cos"][(1, 1)], -0.2)
            self.assertEqual(parameters["X1pert_b_cos"][(2, 2)], 0.3)

    def test_parameter_write_failure_obeys_cleanup_policy(self):
        eq = Gvec(keep_failures=False)
        with ScratchDir("/tmp"), patch(
            "simsopt.mhd.gvec.gvec.util.write_parameters",
            side_effect=OSError("write failed"),
        ):
            with self.assertRaisesRegex(OSError, "write failed"):
                eq.run()
            failed_directory = Path.cwd() / eq.rundir
            self.assertFalse(failed_directory.exists())

    def test_non_gvec_run_exception_is_cleaned_and_preserved(self):
        eq = Gvec(keep_failures=False)
        with ScratchDir("/tmp"), patch(
            "simsopt.mhd.gvec.gvec.run",
            side_effect=ValueError("invalid run configuration"),
        ) as run, patch("simsopt.mhd.gvec.gvec.util.write_parameters"):
            with self.assertRaisesRegex(ValueError, "invalid run configuration"):
                eq.run()
            self.assertFalse(eq.rundir.exists())
            with self.assertRaisesRegex(ValueError, "invalid run configuration"):
                eq.run()
            self.assertFalse(eq.rundir.exists())
        self.assertEqual(run.call_count, 2)

    def test_state_acquisition_failure_is_cleaned_and_preserved(self):
        eq = Gvec(keep_failures=False)
        with ScratchDir("/tmp"), patch(
            "simsopt.mhd.gvec.gvec.run",
            return_value=_StateFailureRun(),
        ), patch("simsopt.mhd.gvec.gvec.util.write_parameters"):
            with self.assertRaisesRegex(ValueError, "state unavailable"):
                eq.run()
            self.assertFalse(eq.rundir.exists())

        self.assertFalse(eq.run_successful)
        self.assertIsNone(eq._state)
