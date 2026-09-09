import math
import unittest
from collect_metrics import relative_metrics, validated_drr, TASKS


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

    def test_drr_requires_complete_checkpoint_gates_and_counts(self):
        report = dict(status='complete', seed=42, steps_per_task=10000,
            checks=[dict(method=m, after_task=t, global_step=True, num_seen_examples=True,
                         ordered_buffer_pair_names=True, numpy_rng=True, all_batch_streams=True)
                    for m in ('mer', 'samcl') for t in TASKS],
            task_records=[dict(method=m, source_task=t, unique_later_replayed_pairs=2,
                               training_pairs=10, DRR_task=.2)
                          for m in ('mer', 'samcl') for t in TASKS[:-1]],
            methods=[dict(method=m, DRR=.2, tasks=3) for m in ('mer', 'samcl')])
        self.assertEqual(validated_drr(report), {'mer': .2, 'samcl': .2})
        report['checks'][0]['numpy_rng'] = False
        with self.assertRaises(ValueError):
            validated_drr(report)
        report['checks'][0]['numpy_rng'] = True
        report['task_records'][0]['unique_later_replayed_pairs'] = 3
        with self.assertRaises(ValueError):
            validated_drr(report)


if __name__ == '__main__':
    unittest.main()
