"""比較条件の時間境界と、欠測日を含む16暦日の扱いを確認する。"""
import unittest
import pandas as pd
from backtest import select_period


class BacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dates = pd.date_range('2013-01-01', '2017-08-15')
        cls.data = pd.DataFrame({'date': dates, 'id': range(len(dates)),
                                 'store_nbr': 1, 'family': 'A', 'sales': 1.0, 'onpromotion': 0})

    def test_train_and_validation_never_overlap(self):
        for policy in ['era_expanding','recent_365_days']:
            history, train, valid = select_period(self.data, '2017-07-31', policy)
            self.assertLess(history.date.max(), valid.date.min())
            self.assertEqual(valid.date.nunique(), 16)
            self.assertGreaterEqual(train.date.min(), pd.Timestamp('2016-01-01'))
            if policy == 'recent_365_days':
                self.assertGreaterEqual(train.date.min(), pd.Timestamp('2016-07-31'))

    def test_selected_eras_do_not_train_on_2015(self):
        for start in ['2016-08-01','2014-08-01']:
            history, _, _ = select_period(self.data, start, 'era_expanding')
            self.assertNotIn(2015, history.date.dt.year.unique())

    def test_missing_date_does_not_extend_horizon(self):
        data = self.data.loc[self.data.date != '2014-12-25']
        _, _, valid = select_period(data, '2014-12-16', 'era_expanding')
        self.assertEqual(valid.date.nunique(), 15)
        self.assertEqual(valid.date.max(), pd.Timestamp('2014-12-31'))
