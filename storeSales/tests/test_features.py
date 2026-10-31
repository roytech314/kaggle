"""日付整合・予測起点の制約・系列の分離を実データに依存せず確認する。"""
import unittest
import numpy as np
import pandas as pd
from features import build_features
from evaluation import split_last_days


class FeatureTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2020-01-01', periods=90)
        self.history = pd.concat([
            pd.DataFrame({'date': dates, 'store_nbr': store, 'family': 'A',
                          'sales': np.arange(90) + store * 100, 'onpromotion': 0})
            for store in [1, 2]
        ], ignore_index=True)

    def test_future_order_and_cutoff(self):
        future = pd.DataFrame({'date': [pd.Timestamp('2020-04-15'), pd.Timestamp('2020-03-31')],
                               'store_nbr': [2, 1], 'family': 'A', 'onpromotion': 0})
        x = build_features(self.history, future)
        np.testing.assert_array_equal(x.lag_16, [289, 174])
        np.testing.assert_allclose(x.rolling_mean_7_shift_16, [286, 171])
        # 未来のsales列に何を入れても結果は変わらない。
        pd.testing.assert_frame_equal(x, build_features(self.history, future.assign(sales=999999)))

    def test_missing_date_is_not_previous_row(self):
        h = self.history.loc[self.history.date != '2020-03-01']
        target = self.history.loc[(self.history.date == '2020-03-17') & (self.history.store_nbr == 1)]
        self.assertTrue(np.isnan(build_features(h, target).lag_16.iloc[0]))

    def test_newer_sales_cannot_affect_features(self):
        target = self.history.loc[self.history.date == '2020-03-20']
        changed = self.history.copy()
        changed.loc[changed.date > '2020-03-04', 'sales'] = 999999
        pd.testing.assert_frame_equal(build_features(self.history, target), build_features(changed, target))

    def test_split_keeps_dates_together(self):
        fit, valid = split_last_days(self.history)
        self.assertLess(fit.date.max(), valid.date.min())
        self.assertEqual(valid.date.nunique(), 16)
        self.assertEqual(len(valid), 32)


if __name__ == '__main__':
    unittest.main()
