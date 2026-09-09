import math
import unittest
from collect_metrics import relative_metrics


class MetricsTest(unittest.TestCase):
    def test_dice_retention_and_learning(self):
        b, r = relative_metrics(.8, .6, .5)
        self.assertAlmostEqual(b, -.25); self.assertAlmostEqual(r, 1.6)

    def test_tre_direction_and_unit_invariance(self):
        a = relative_metrics(4., 5., 3., lower=True)
        b = relative_metrics(4000., 5000., 3000., lower=True)
        self.assertEqual(a, b); self.assertAlmostEqual(a[0], -.2); self.assertEqual(a[1], .75)

    def test_missing_reference_is_not_zero(self):
        self.assertEqual(relative_metrics(.5, .5), (0., None))

    def test_invalid_denominators(self):
        for value in (0., -1., math.nan, math.inf):
            with self.assertRaises(ValueError):
                relative_metrics(value, 1.)


if __name__ == '__main__':
    unittest.main()
