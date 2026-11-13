from dataclasses import replace
import json
import math
import shutil
import subprocess
import unittest
import xml.etree.ElementTree as ET

from experiments.spectral_operator_fft_selection.run import (
    Budget, circular_convolution, dft, direct_dft2, estimate, fft2, fixture,
    full_spectrum_inverse, maximum_error, outputs, rfft2, shape_of, simulate,
    spectrum_product, staged_inverse,
)


class SpectralOperatorFftTests(unittest.TestCase):
    def test_separable_transforms_match_independent_direct_and_round_trip(self):
        for nx,ny in ((3,5),(4,6),(2,3),(1,5),(3,1)):
            grid,_ = fixture(nx,ny)
            for norm in ("forward","backward","ortho"):
                separated = fft2(grid,norm=norm)
                self.assertLess(maximum_error(separated,direct_dft2(grid,norm=norm)),1e-12)
                self.assertLess(maximum_error(fft2(separated,True,norm),grid),1e-12)
                one_sided = rfft2(grid,norm)
                self.assertEqual(shape_of(one_sided),(nx,ny//2+1))
                self.assertLess(maximum_error(staged_inverse(one_sided,(nx,ny),norm),grid),1e-12)

    def test_spectral_multiplication_matches_independent_spatial_circular_convolution(self):
        for shape in ((3,5),(4,6)):
            grid,kernel = fixture(*shape)
            reference = circular_convolution(grid,kernel)
            for norm in ("forward","backward","ortho"):
                spectrum = spectrum_product(grid,kernel,norm)
                self.assertLess(maximum_error(staged_inverse(spectrum,shape,norm),reference),1e-12)
                self.assertLess(maximum_error(full_spectrum_inverse(spectrum,shape,norm),reference),1e-12)

    def test_raw_boundary_columns_keep_valid_row_phase_until_first_inverse(self):
        grid,kernel = fixture(4,6)
        spectrum = spectrum_product(grid,kernel)
        self.assertGreater(max(abs(row[0].imag) for row in spectrum),0.01)
        self.assertGreater(max(abs(row[-1].imag) for row in spectrum),0.01)
        reference = circular_convolution(grid,kernel)
        wrong = staged_inverse(spectrum,(4,6),premature_clear=True)
        self.assertGreater(maximum_error(wrong,reference),0.1)
        columns = [dft([row[ky] for row in spectrum],True) for ky in (0,3)]
        self.assertLess(max(abs(v.imag) for column in columns for v in column),1e-12)

    def test_wrong_odd_nyquist_pair_factor_axis_halving_and_shape_controls(self):
        cases = {row["id"]:row for row in simulate()["cases"]}
        self.assertTrue(cases["staged-real-inverse"]["matches_reference"])
        shape = cases["implicit-even-inverse-length"]
        self.assertEqual(shape["output_shape"],(3,4))
        self.assertIsNone(shape["maximum_absolute_error"])
        self.assertFalse(shape["matches_reference"])
        for name in ("premature-frequency-column-clear","odd-last-bin-treated-as-nyquist",
                     "missing-conjugate-pair-factor","incorrectly-half-every-axis"):
            self.assertFalse(cases[name]["matches_reference"])
            self.assertGreater(cases[name]["maximum_absolute_error"],0.05)

    def test_paired_norms_preserve_amplitude_and_truncation_has_its_own_reference(self):
        report = simulate()
        for row in report["normalization_checks"]:
            self.assertLess(row["maximum_absolute_error"],1e-12)
        grid,kernel = fixture()
        reference = circular_convolution(grid,kernel)
        wrong = staged_inverse(spectrum_product(grid,kernel),(3,5),"backward")
        self.assertLess(maximum_error(wrong,[[v/15 for v in row] for row in reference]),1e-12)
        self.assertGreater(report["mode_truncation"]["full_operator_difference"],0.4)
        self.assertLess(report["mode_truncation"]["same_retained_spectrum_reference_error"],1e-12)
        self.assertIsNone(report["mode_truncation"]["pde_quality"])

    def test_dense_complex_budget_odd_dimensions_and_explicit_invalid_inputs(self):
        result = estimate(Budget())
        self.assertEqual(result["last_axis_bins"],65)
        self.assertEqual(result["retained_last_axis_bins"],17)
        self.assertEqual(result["dense_weight_bytes"],32*64*32*17*8)
        self.assertEqual(result["input_output_spectrum_bytes"],4*96*128*65*8)
        odd = estimate(replace(Budget(),grid_x=127,grid_y=125,modes_x=31,modes_y=25))
        self.assertEqual(odd["last_axis_bins"],63)
        self.assertEqual(odd["retained_last_axis_bins"],13)
        self.assertIsNone(odd["peak_hbm_bytes"])
        for profile in (replace(Budget(),grid_y=0),replace(Budget(),modes_y=129),
                        replace(Budget(),real_scalar_bytes=2),replace(Budget(),assumed_read_gbps=math.nan)):
            with self.assertRaises(ValueError):
                estimate(profile)
        for invalid in ([[1],[2,3]], [[math.inf]]):
            with self.assertRaises(ValueError):
                rfft2(invalid)
        with self.assertRaises(ValueError):
            dft([1,2],norm="none")
        with self.assertRaises(ValueError):
            staged_inverse([[1j,2j]],(2,3))

    def test_artifacts_are_reproducible_seven_cards_and_safe_svg_bounds(self):
        for path,content in outputs().items():
            self.assertEqual(path.read_text(encoding="utf-8"),content)
            if path.name.endswith("-cards.html"):
                self.assertEqual(content.count('<section class="card"'),7)
                self.assertNotIn("<script",content)
                self.assertIn("default-src 'none'; style-src 'unsafe-inline'",content)
            if path.suffix == ".svg":
                root = ET.fromstring(content)
                self.assertEqual(root.attrib["viewBox"],"0 0 1600 1000")
                for polygon in root.iter("{http://www.w3.org/2000/svg}polygon"):
                    self.assertTrue(all(200 <= float(point.split(",")[1]) <= 620
                                        for point in polygon.attrib["points"].split()))

    @unittest.skipUnless(shutil.which("node"),"Node.js optional")
    def test_javascript_budget_parity_and_overflow_rejection(self):
        from experiments.spectral_operator_fft_selection.render import CALCULATOR
        profiles = [Budget(),replace(Budget(),grid_y=125,modes_y=25),
                    replace(Budget(),batch=1,real_scalar_bytes=8)]
        source = CALCULATOR.split("const form=")[0]
        script = source+"\nconsole.log(JSON.stringify("+json.dumps([estimate(p)["input"] for p in profiles])+".map(estimateBudget)));"
        actual = json.loads(subprocess.run(["node","-e",script],check=True,capture_output=True,text=True).stdout)
        for profile,result in zip(profiles,actual):
            for key,value in result.items():
                if value is None:
                    self.assertIsNone(estimate(profile)[key])
                else:
                    self.assertAlmostEqual(value,estimate(profile)[key],places=9)
        huge = {**estimate(Budget())["input"],"batch":9007199254740991}
        script = source+"\ntry{estimateBudget("+json.dumps(huge)+");process.exit(1)}catch(e){console.log(e.message)}"
        self.assertIn("exact numerical range",subprocess.run(
            ["node","-e",script],check=True,capture_output=True,text=True).stdout)


if __name__ == "__main__":
    unittest.main()
