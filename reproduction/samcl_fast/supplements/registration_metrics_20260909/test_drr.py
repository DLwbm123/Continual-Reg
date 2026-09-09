"""Run inside the existing experiment environment: python -m unittest test_drr."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from reconstruct_drr import SamplingOnly, noop, verify_stage


class DrrTests(unittest.TestCase):
    def test_actual_consumption_unique_historical_pairs(self):
        model = SamplingOnly.__new__(SamplingOnly)
        model.origin = {'old': 'oasis', 'current': 'ctct', 'future': 'nlst'}
        model.task_index = 1
        model.historical = {'oasis': set()}
        model.stage_used = {'oasis': set()}
        model.forward({'names': ['old', 'current', 'old']})
        model.forward({'names': ['old']})
        self.assertEqual(model.historical, {'oasis': {'old'}})
        with self.assertRaises(ValueError):
            model.forward({'names': ['future']})

    def test_original_samcl_discards_first_draw(self):
        cfg = SimpleNamespace(
            method=SimpleNamespace(buffer=SimpleNamespace(cpu=True), mer=SimpleNamespace(beta=.25)),
            exp=SimpleNamespace(train=SimpleNamespace(batch_size=4)))
        model = SamplingOnly('samcl', cfg, {'unused': 'oasis', 'used': 'oasis', 'current': 'ctct'})
        model.task_index = 1
        draws = iter([{'names': ['unused']}, {'names': ['used']}])
        model.buffer = SimpleNamespace(is_empty=lambda: False, add_data_dict=noop,
                                       get_data_dict=lambda: next(draws))
        model.observe({'names': ['current'] * 4, 'imgs': torch.zeros(4, 1)})
        self.assertEqual(model.historical['oasis'], {'used'})
        self.assertEqual(list(draws), [])

    def test_checkpoint_gate_rejects_wrong_buffer_and_rng(self):
        model = SimpleNamespace(buffer=SimpleNamespace(num_seen_examples=4, buffer_data=[{'names': 'pair'}]))
        payload = dict(global_step=1, replay_buffer=dict(num_seen_examples=4, buffer_data=[{'names': 'pair'}]),
                       rng_states={'numpy': np.random.get_state()}, stream_states={})
        self.assertTrue(all(verify_stage(payload, model, {}, 1).values()))
        np.random.random()
        with self.assertRaises(ValueError):
            verify_stage(payload, model, {}, 1)
        payload['rng_states']['numpy'] = np.random.get_state()
        payload['replay_buffer']['buffer_data'][0]['names'] = 'wrong'
        with self.assertRaises(ValueError):
            verify_stage(payload, model, {}, 1)


if __name__ == '__main__':
    unittest.main()
