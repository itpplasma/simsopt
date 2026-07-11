import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xarray as xr

from simsopt._core.optimizable import Optimizable
from simsopt.mhd.gvec_quantities import (
    GvecQuasisymmetryRatioResidual,
    _quasisymmetry_result,
)

try:
    import gvec
except ImportError:
    gvec = None

TEST_DIR = Path(__file__).parent / ".." / "test_files"


def quasisymmetry_dataset():
    coordinates = {
        "rad": np.array([0.25, 0.75]),
        "pol": np.array([0.0, np.pi]),
        "tor": np.array([0.0, np.pi]),
        "xyz": np.arange(3),
    }
    shape = (2, 2, 2)
    vectors = shape + (3,)
    B = np.zeros(vectors)
    B[..., 1] = 1.0
    grad_mod_B = np.zeros(vectors)
    grad_mod_B[..., 2] = 1.0
    grad_rho = np.zeros(vectors)
    grad_rho[..., 0] = 1.0
    dataset = xr.Dataset(coords=coordinates)
    for name, value in {
        "B": B,
        "grad_mod_B": grad_mod_B,
        "grad_rho": grad_rho,
    }.items():
        dataset[name] = (("rad", "pol", "tor", "xyz"), value)
    for name, value in {"mod_B": 2.0, "Jac": 1.0}.items():
        dataset[name] = (("rad", "pol", "tor"), np.full(shape, value))
    for name, value in {
        "dPhi_dr": 1.0,
        "iota": 0.4,
        "B_theta_avg": 0.0,
        "B_zeta_avg": 1.0,
    }.items():
        dataset[name] = (("rad",), np.full(2, value))
    return dataset


class GvecQuasisymmetryUnitTests(unittest.TestCase):
    def test_public_mhd_export(self):
        from simsopt.mhd import GvecQuasisymmetryRatioResidual as PublicResidual

        self.assertIs(PublicResidual, GvecQuasisymmetryRatioResidual)

    def test_maps_flux_coordinate_and_vmec_helicity_without_boundary_metadata(self):
        observed = {}

        class FakeState:
            nfp = 4

            def evaluate(self, *quantities, **coordinates):
                observed["quantities"] = quantities
                observed["coordinates"] = coordinates
                return quasisymmetry_dataset()

        eq = Optimizable()
        eq.state = FakeState()
        residual = GvecQuasisymmetryRatioResidual(
            eq,
            [0.0, 0.25],
            helicity_m=1,
            helicity_n=-1,
            ntheta=2,
            nphi=2,
        )

        result = residual.compute()

        self.assertEqual(result.native_helicity_n, 4)
        np.testing.assert_allclose(result.rho, [1.0e-4, 0.5])
        np.testing.assert_allclose(observed["coordinates"]["zeta"], [0.0, np.pi / 4])
        self.assertIn("B_zeta_avg", observed["quantities"])

    def test_backend_neutral_residual_algebra_and_layout(self):
        result = _quasisymmetry_result(
            quasisymmetry_dataset(),
            np.array([1.0, 4.0]),
            helicity_m=1,
            native_helicity_n=4,
            surfaces=np.array([0.0625, 0.5625]),
            rho=np.array([0.25, 0.75]),
        )

        expected = np.concatenate((np.full(4, 0.225), np.full(4, 0.45)))
        np.testing.assert_allclose(result.residuals1d, expected)
        np.testing.assert_allclose(result.profile, [0.2025, 0.81])
        self.assertAlmostEqual(result.total, 1.0125)
        self.assertEqual(result.residuals3d.shape, (2, 2, 2))

    def test_rejects_nonpositive_jacobian(self):
        dataset = quasisymmetry_dataset()
        dataset["Jac"][0, 0, 0] = 0.0

        with self.assertRaisesRegex(RuntimeError, "positive finite Jacobian"):
            _quasisymmetry_result(
                dataset,
                np.ones(2),
                helicity_m=1,
                native_helicity_n=4,
                surfaces=np.array([0.0, 1.0]),
                rho=np.array([1.0e-4, 1.0]),
            )

    def test_validates_surface_weight_and_grid_contracts(self):
        eq = Optimizable()
        invalid = [
            {"surfaces": []},
            {"surfaces": [-0.1]},
            {"surfaces": [0.5], "weights": [-1.0]},
            {"surfaces": [0.5], "weights": [1.0, 2.0]},
            {"surfaces": [0.5], "ntheta": 0},
            {"surfaces": [0.5], "nphi": 1.5},
            {"surfaces": [0.5], "helicity_m": 0, "helicity_n": 0},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                GvecQuasisymmetryRatioResidual(eq, **kwargs)


@unittest.skipIf(gvec is None, "gvec package not installed")
class GvecQuasisymmetryIntegrationTests(unittest.TestCase):
    def test_vmec_import_preserves_quasisymmetry_residual(self):
        from gvec.scripts.convert_wout import convert_vmec_wout
        from simsopt.mhd import QuasisymmetryRatioResidual, Vmec

        wout = TEST_DIR / "wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
        surfaces = np.linspace(0.0, 1.0, 11)
        with tempfile.TemporaryDirectory() as directory:
            convert_vmec_wout(
                wout,
                Path(directory),
                extra_parameters={
                    "sgrid_nElems": 25,
                    "sgrid_grid_type": 4,
                    "X1X2_deg": 5,
                    "LA_deg": 5,
                },
            )
            eq = Optimizable()
            eq.state = gvec.find_state(directory)
            try:
                eq.boundary = SimpleNamespace(nfp=4)
                gvec_residual = GvecQuasisymmetryRatioResidual(
                    eq, surfaces, helicity_m=1, helicity_n=0
                )
                vmec_residual = QuasisymmetryRatioResidual(
                    Vmec(str(wout)), surfaces, helicity_m=1, helicity_n=0
                )
                relative_error = abs(
                    gvec_residual.total() / vmec_residual.total() - 1.0
                )
                residual_count = gvec_residual.residuals().size
            finally:
                eq.state.unbind()

        self.assertLess(relative_error, 1.0e-4)
        self.assertEqual(residual_count, 11 * 63 * 64)
