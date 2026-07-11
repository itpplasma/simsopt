import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from monty.tempfile import ScratchDir

from simsopt._core.optimizable import Optimizable
from simsopt.geo import Surface, SurfaceRZFourier
from simsopt.mhd.profiles import Profile, ProfileScaled, ProfilePolynomial, ProfileSpline
from simsopt.mhd.gvec import Gvec, GVECSurfaceDoFs

try:
    import gvec
except ImportError:
    gvec = None

TEST_DIR = Path(__file__).parent / ".." / "test_files"


def axisymmetric_surface():
    surface = SurfaceRZFourier(nfp=1, mpol=1, ntor=0, stellsym=True)
    surface.set_rc(0, 0, 5.0)
    surface.set_rc(1, 0, 1.0)
    surface.set_zs(1, 0, 1.0)
    return surface


def axisymmetric_parameters(project_name):
    return {
        "ProjectName": project_name,
        "X1X2_deg": 3,
        "LA_deg": 3,
        "totalIter": 2000,
        "minimize_tol": 1.0e-4,
        "sgrid": {"grid_type": 4, "nElems": 2},
    }

@unittest.skipIf(gvec is None, "gvec package not installed")
class GvecTests(unittest.TestCase):

    # === helpers === #

    def check_Optimizable(self, eq):
        """assert that eq is a properly configured optimizable"""
        self.assertIsInstance(eq, Gvec)
        self.assertIsInstance(eq, Optimizable)

        self.assertListEqual(eq.local_full_dof_names, ["phiedge"])
        self.assertEqual(eq.local_full_x.size, 1)

        self.assertEqual(len(eq.parents), 3)
        self.assertIn(eq.boundary, eq.parents)
        self.assertIn(eq.pressure_profile, eq.parents)
        if eq.current_profile is None:
            self.assertIn(eq.iota_profile, eq.parents)
        else:
            self.assertIn(eq.current_profile, eq.parents)
            # with prescribed current, iota_profile is only used as an initial condition
            # with only 3 parents, we don't need to check explicitly

        self.assertIsInstance(eq.boundary, (Surface, GVECSurfaceDoFs))
        self.assertIsInstance(eq.pressure_profile, Profile)
        if eq.iota_profile is not None:
            self.assertIsInstance(eq.iota_profile, Profile)
        if eq.current_profile is not None:
            self.assertIsInstance(eq.current_profile, Profile)

    def check_consistency(self, eq):
        """assert that eq is internally consistent"""
        # check consistency: phiedge
        #   In GVEC the input 'phiedge' is the flux through the cross-section,
        #   but the 'Phi' profile refers to the component of the vector potential.
        #   These are related by a factor 2π.
        phiedge = eq.state.evaluate("Phi", rho=1.0)["Phi"].item() * 2 * np.pi
        self.assertEqual(eq.phiedge, phiedge)

        # check consistency: pressure profile
        rho = np.linspace(0, 1, 11)
        pressure = eq.state.evaluate("p", rho=rho).p
        np.testing.assert_allclose(eq.pressure_profile(rho**2), pressure)

        # check consistency: rotational transform profile (iota)
        if eq.current_profile is None:
            iota = eq.state.evaluate("iota", rho=rho).iota
            np.testing.assert_allclose(eq.iota_profile(rho**2), iota)

        # check consistency: current profile
        # absolute tolerance of 10kA
        if eq.current_profile is not None:
            # np.testing.assert_allclose(eq.state.evaluate("iota_curr", rho=rho).iota_curr, 0.0)
            I_tor = eq.state.evaluate("I_tor", rho=rho).I_tor
            np.testing.assert_allclose(eq.current_profile(rho**2), I_tor, atol=1e4)

        # check consistency: boundary
        if isinstance(eq.boundary, SurfaceRZFourier):
            theta = eq.boundary.quadpoints_theta * 2 * np.pi
            zeta = -eq.boundary.quadpoints_phi * 2 * np.pi
            boundary = eq.state.evaluate("pos", rho=1.0, theta=theta, zeta=zeta)["pos"].squeeze().transpose("tor", "pol", "xyz")
            np.testing.assert_allclose(eq.boundary.gamma(), boundary)

            volume = eq.state.evaluate("V").V.item()
            self.assertAlmostEqual(eq.boundary.volume(), volume)

    def check_return_functions(self, eq):
        """check that the return functions work and return a float"""
        run_count0 = eq.run_count
        self.assertFalse(eq.run_required)

        self.assertIsInstance(eq.aspect(), float)
        self.assertIsInstance(eq.volume(), float)
        self.assertIsInstance(eq.iota_axis(), float)
        self.assertIsInstance(eq.iota_edge(), float)
        self.assertIsInstance(eq.mean_iota(), float)
        self.assertIsInstance(eq.mean_shear(), float)
        self.assertIsInstance(eq.vacuum_well(), float)

        # no new run should have been triggered
        self.assertEqual(eq.run_count, run_count0)

    # === tests === #

    def test_init_defaults(self):
        eq = Gvec()
        self.check_Optimizable(eq)
        self.assertTrue(eq.run_required)

        self.assertIsInstance(eq.boundary, SurfaceRZFourier)
        reference_surf = SurfaceRZFourier()
        self.assertEqual(eq.boundary.nfp, reference_surf.nfp)
        self.assertEqual(eq.boundary.stellsym, reference_surf.stellsym)
        self.assertEqual(eq.boundary.mpol, reference_surf.mpol)
        self.assertEqual(eq.boundary.ntor, reference_surf.ntor)
        np.testing.assert_allclose(eq.boundary.x, reference_surf.x)

        self.assertIsInstance(eq.pressure_profile, ProfilePolynomial)
        np.testing.assert_equal(eq.pressure_profile.local_full_x, [0.0])

        self.assertIsInstance(eq.current_profile, ProfilePolynomial)
        np.testing.assert_equal(eq.current_profile.local_full_x, [0.0])

        self.assertEqual(eq.phiedge, 1.0)

    def test_init_from_parameter_file(self):
        eq = Gvec.from_parameter_file(TEST_DIR / "parameter-LandremanPaul2021_QA.gvec.toml")
        self.check_Optimizable(eq)
        self.assertTrue(eq.run_required)

        self.assertIsInstance(eq.boundary, SurfaceRZFourier)
        self.assertEqual(eq.boundary.nfp, 2)
        self.assertEqual(eq.boundary.stellsym, True)
        self.assertEqual(eq.boundary.mpol, 15)
        self.assertEqual(eq.boundary.ntor, 12)
        self.assertAlmostEqual(eq.boundary.get("rc(2,-4)"), 4.850684989433037e-05)

        self.assertIsInstance(eq.pressure_profile, ProfileScaled)
        self.assertIsInstance(eq.pressure_profile.base, ProfilePolynomial)
        np.testing.assert_equal(eq.pressure_profile.base.local_full_x, [0.0])

        self.assertIsInstance(eq.current_profile, ProfileScaled)
        self.assertIsInstance(eq.current_profile.base, ProfilePolynomial)
        np.testing.assert_equal(eq.current_profile.base.local_full_x, [0.0])

        self.assertEqual(eq.phiedge, -0.08385727554)

        self.assertEqual(eq.parameters["sgrid"]["nElems"], 5)

    def test_init_from_rundir(self):
        eq = Gvec.from_rundir(TEST_DIR / "gvec-W7-X_standard_configuration")
        self.check_Optimizable(eq)
        self.assertFalse(eq.run_required)
        self.assertTrue(eq.run_successful)
        loaded_state = eq._state
        with patch("simsopt.mhd.gvec.gvec.run") as run:
            self.assertIs(eq.state, loaded_state)
        run.assert_not_called()
        self.assertEqual(eq.run_count, -1)
        self.check_consistency(eq)
        self.check_return_functions(eq)

        self.assertIsInstance(eq.boundary, SurfaceRZFourier)
        self.assertEqual(eq.boundary.nfp, 5)
        self.assertEqual(eq.boundary.stellsym, True)
        self.assertEqual(eq.boundary.mpol, 11)
        self.assertEqual(eq.boundary.ntor, 12)
        self.assertAlmostEqual(eq.boundary.get("rc(2,-4)"), -0.000133285510379407)

        self.assertIsInstance(eq.pressure_profile, ProfileScaled)
        self.assertAlmostEqual(eq.pressure_profile.local_full_x[0], 1.0)
        self.assertIsInstance(eq.pressure_profile.base, ProfilePolynomial)
        np.testing.assert_equal(eq.pressure_profile.base.local_full_x, [1e-6, -1e-6])

        self.assertIsInstance(eq.current_profile, ProfileScaled)
        self.assertIsInstance(eq.current_profile.base, ProfilePolynomial)
        np.testing.assert_equal(eq.current_profile.base.local_full_x, [0.0])

        self.assertEqual(eq.phiedge, 2.1907427)

        self.assertEqual(eq.parameters["sgrid"]["nElems"], 5)
        self.assertEqual(eq.parameters["X1X2_deg"], 5)
        self.assertEqual(eq.parameters["LA_deg"], 5)

    def test_from_rundir_rejects_equilibrium_overrides(self):
        with self.assertRaisesRegex(ValueError, "equilibrium overrides: boundary"):
            Gvec.from_rundir(
                TEST_DIR / "gvec-W7-X_standard_configuration",
                boundary=SurfaceRZFourier(),
            )

    def test_run_from_rundir(self):
        with ScratchDir("/tmp"):
            eq = Gvec.from_rundir(TEST_DIR / "gvec-W7-X_standard_configuration")
            self.check_Optimizable(eq)
            self.assertFalse(eq.run_required)
            self.assertTrue(eq.run_successful)
            self.check_consistency(eq)
            self.check_return_functions(eq)

            eq.parameters["minimize_tol"] = 1e-3
            eq.run(force=True)
            self.assertFalse(eq.run_required)
            self.assertTrue(eq.run_successful)
            self.check_consistency(eq)
            self.check_return_functions(eq)

    def check_pressure_profile(self, profile, pressure_on_axis, profile_type):
        with ScratchDir("/tmp"):
            eq = Gvec.from_parameter_file(
                TEST_DIR / "parameter-LandremanPaul2021_QA_lowres.gvec.toml",
                require_convergence=False,
            )
            eq.parameters["totalIter"] = 10
            eq.pressure_profile = profile
            self.assertEqual(eq.pressure_profile, profile)
            self.assertTrue(eq.run_required)
            eq.run()
            self.assertTrue(eq.run_successful)
            self.check_consistency(eq)
            self.check_return_functions(eq)
            self.assertEqual(eq.state.parameters["pres"]["type"], profile_type)
            p_axis = eq.state.evaluate("p", rho=0.0).p.item()
            self.assertAlmostEqual(p_axis, pressure_on_axis)

    def test_set_polynomial_pressure_profile(self):
        profile = ProfilePolynomial(1.0e2 * np.array([1, 1, -2.0]))
        self.check_pressure_profile(profile, 1.0e2, "polynomial")

    def test_set_scaled_pressure_profile(self):
        profile = ProfileScaled(ProfilePolynomial([1, 1, -2.0]), 1.5e2)
        self.check_pressure_profile(profile, 1.5e2, "polynomial")

    def test_set_spline_pressure_profile(self):
        s_spline = np.linspace(0, 1, 5)
        profile = ProfileSpline(
            s_spline,
            1.0e2 * (2.0 + 0.6 * s_spline - 1.5 * s_spline**2),
        )
        self.check_pressure_profile(profile, 2.0e2, "interpolation")

    def check_iota_profile(self, profile, expected_type):
        eq = Gvec.from_parameter_file(
            TEST_DIR / "parameter-LandremanPaul2021_QA_lowres.gvec.toml"
        )
        eq.iota_profile = profile
        eq.current_profile = None

        parameters = eq.prepare_parameters()
        iota = parameters["iota"]

        self.assertEqual(iota["type"], expected_type)
        self.assertNotIn("I_tor", parameters)
        self.assertNotIn("picard_current", parameters)
        if expected_type == "polynomial":
            s = np.linspace(0.0, 1.0, 11)
            represented = np.polynomial.polynomial.polyval(s, iota["coefs"])
        else:
            s = np.asarray(iota["rho2"])
            represented = np.asarray(iota["vals"])
        np.testing.assert_allclose(represented, profile(s))

    def test_set_polynomial_iota_profile(self):
        self.check_iota_profile(ProfilePolynomial(np.array([1, 1, -2.0])), "polynomial")

    def test_set_spline_iota_profile(self):
        s_spline = np.linspace(0, 1, 5)
        profile = ProfileSpline(s_spline, 2.0 + 0.6 * s_spline - 1.5 * s_spline**2)
        self.check_iota_profile(profile, "interpolation")

    def test_iota_profile_native_run_converges(self):
        profile = ProfilePolynomial([0.05, 0.02])
        eq = Gvec(
            phiedge=1.0,
            boundary=axisymmetric_surface(),
            pressure=ProfilePolynomial([0.0]),
            iota=profile,
            current=None,
            parameters=axisymmetric_parameters("axisym_fixed_iota"),
            require_convergence=True,
        )

        with ScratchDir("/tmp"):
            eq.run()
            iota = eq.state.evaluate(
                "iota", rho=np.array([0.25, 0.5, 0.75, 1.0])
            ).iota

        self.assertTrue(eq.run_successful)
        self.assertLessEqual(eq._runobj.max_force, 1.0e-4)
        rho = np.array([0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(iota, profile(rho**2), atol=1.0e-12)

    def check_current_profile(self, profile, expected_type):
        eq = Gvec.from_parameter_file(
            TEST_DIR / "parameter-LandremanPaul2021_QA_lowres.gvec.toml"
        )
        eq.current_profile = profile
        eq.iota_profile = None

        parameters = eq.prepare_parameters()
        current = parameters["I_tor"]

        self.assertEqual(current["type"], expected_type)
        self.assertEqual(parameters["picard_current"], "auto")
        self.assertNotIn("iota", parameters)
        if expected_type == "polynomial":
            s = np.linspace(0.0, 1.0, 11)
            represented = np.polynomial.polynomial.polyval(s, current["coefs"])
            represented *= current.get("scale", 1.0)
        else:
            s = np.asarray(current["rho2"])
            represented = np.asarray(current["vals"])
        np.testing.assert_allclose(represented, profile(s))

    def test_set_polynomial_current_profile(self):
        profile = ProfilePolynomial(1.0e4 * np.array([0, 1.1, -0.1]))
        self.check_current_profile(profile, "polynomial")

    def test_set_scaled_current_profile(self):
        profile = ProfileScaled(ProfilePolynomial(np.array([0, 1.1, -0.1])), 1.0e4)
        self.check_current_profile(profile, "polynomial")

    def test_set_spline_current_profile(self):
        s_spline = np.linspace(0, 1, 5)
        profile = ProfileSpline(s_spline, 1.0e4 * (1.1 * s_spline - 0.1 * s_spline**2))
        self.check_current_profile(profile, "interpolation")

    def test_set_scaled_spline_current_profile(self):
        s_spline = np.linspace(0, 1, 5)
        profile = ProfileScaled(
            ProfileSpline(s_spline, 1.1 * s_spline - 0.1 * s_spline**2),
            1.0e4,
        )
        self.check_current_profile(profile, "interpolation")

    def test_explicit_current_stages_are_preserved(self):
        stages = [
            {"picard_current": {"iota_tol": 1.0e-5, "target": "iota"}},
            {"picard_current": "off", "minimize_tol": 1.0e-4},
        ]
        eq = Gvec(current=ProfilePolynomial([0.0]), parameters={"stages": stages})

        parameters = eq.prepare_parameters()

        self.assertEqual(parameters["stages"], stages)
        self.assertEqual(parameters["picard_current"], {})

    def test_explicit_current_control_is_preserved(self):
        current_control = {"iota_tol": 2.0e-6, "target": "iota_and_force"}
        eq = Gvec(
            current=ProfilePolynomial([0.0]),
            parameters={"picard_current": current_control},
        )

        parameters = eq.prepare_parameters()

        self.assertEqual(parameters["picard_current"], current_control)

    def test_current_mode_excludes_initial_iota_from_dependency_graph(self):
        iota = ProfilePolynomial([0.005])
        current = ProfilePolynomial([0.0, 1000.0])
        eq = Gvec(iota=iota, current=current)

        self.assertEqual(len(eq.parents), 3)
        self.assertIn(current, eq.parents)
        self.assertNotIn(iota, eq.parents)
        eq.run_required = False
        iota.local_full_x = [0.006]
        self.assertFalse(eq.run_required)
        current.local_full_x = [0.0, 900.0]
        self.assertTrue(eq.run_required)

        eq.current_profile = None
        self.assertIn(iota, eq.parents)
        self.assertNotIn(current, eq.parents)
        eq.current_profile = current
        self.assertIn(current, eq.parents)
        self.assertNotIn(iota, eq.parents)

    def test_current_profile_native_run_converges(self):
        profile = ProfileScaled(ProfilePolynomial([0.0, 1.1, -0.1]), 1000.0)
        parameters = axisymmetric_parameters("axisym_scaled_current")
        parameters["picard_current"] = {
            "iota_tol": 1.0e-5,
            "target": "iota_and_force",
        }
        parameters["stages"] = [
            {
                "minimize_tol": 1.0e-3,
                "maxIter": 10,
                "picard_current": {"iota_tol": 1.0e-3, "target": "iota"},
            },
            {
                "minimize_tol": 1.0e-4,
                "picard_current": {
                    "iota_tol": 1.0e-5,
                    "target": "iota_and_force",
                },
            },
            {"minimize_tol": 1.0e-4, "picard_current": "off"},
        ]
        eq = Gvec(
            phiedge=1.0,
            boundary=axisymmetric_surface(),
            pressure=ProfilePolynomial([0.0]),
            iota=ProfilePolynomial([0.005]),
            current=profile,
            parameters=parameters,
            require_convergence=True,
        )

        with ScratchDir("/tmp"):
            eq.run()
            current = eq.state.evaluate(
                "I_tor", rho=np.array([0.25, 0.5, 0.75, 1.0])
            ).I_tor

        self.assertTrue(eq.run_successful)
        self.assertLessEqual(eq._runobj.max_force, 1.0e-4)
        self.assertLessEqual(float(eq._runobj.rms_iota), 1.0e-5)
        self.assertEqual(eq._runobj._state_parameters["picard_current"], "off")
        self.assertEqual(eq._runobj.n_runs_in_stage, [2, 1, 1])
        rho = np.array([0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(current, profile(rho**2), atol=0.05, rtol=1.0e-4)
