"""履歴の保持と改善量の計算を確認する。"""
import csv
import tempfile
import unittest
from pathlib import Path
from experiment_history import record_experiment


class HistoryTests(unittest.TestCase):
    def report(self, run_id='run1'):
        return {'run_id': run_id, 'metric': 'RMSLE',
                'scores': {'baseline': 0.5, 'lightgbm': 0.4},
                'validation_start': '2017-07-31', 'validation_end': '2017-08-15',
                'input_sha256': 'test'}

    def test_records_accumulate_and_delta_uses_own_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            record_experiment(self.report(), tmp)
            second = self.report('run2')
            second['scores'] = {'baseline': 0.8, 'lightgbm': 0.6}
            record_experiment(second, tmp)
            with (Path(tmp) / 'summary.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            self.assertAlmostEqual(float(rows[1]['improvement']), 0.1)
            self.assertAlmostEqual(float(rows[1]['improvement_pct']), 20)
            self.assertAlmostEqual(float(rows[3]['improvement']), 0.2)
            self.assertTrue((Path(tmp) / 'run1.json').exists())

    def test_same_record_is_idempotent_but_changed_score_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            record_experiment(self.report(), tmp)
            record_experiment(self.report(), tmp)
            report = self.report()
            report['scores']['lightgbm'] = 0.3
            with self.assertRaises(ValueError):
                record_experiment(report, tmp)

    def test_invalid_metric_and_nonfinite_score_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.report()
            report['metric'] = 'RMSE'
            with self.assertRaises(ValueError):
                record_experiment(report, tmp)
            report = self.report()
            report['scores']['lightgbm'] = float('nan')
            with self.assertRaises(ValueError):
                record_experiment(report, tmp)
