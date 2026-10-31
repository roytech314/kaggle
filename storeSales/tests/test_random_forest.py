"""RFの前処理が検証データから学習しないことを確認する。"""
import unittest
import numpy as np
import pandas as pd
from random_forest_model import make_preprocessor


class RandomForestPreprocessingTests(unittest.TestCase):
    def test_median_and_categories_are_learned_from_train_only(self):
        train = pd.DataFrame({'store_nbr':[1,2,1], 'family':[0,1,0], 'lag_16':[1.,3.,np.nan]})
        future = pd.DataFrame({'store_nbr':[3,1], 'family':[2,0], 'lag_16':[np.nan,999.]})
        prep = make_preprocessor(train.columns)
        prep.fit(train.astype('float32'))
        result = prep.transform(future.astype('float32'))
        self.assertEqual(result.shape,(2,5))
        np.testing.assert_array_equal(result[0,:4],0)  # 未知の店舗・カテゴリは全0
        self.assertEqual(result[0,-1],2.)  # 学習側[1,3]の中央値
        self.assertEqual(result[1,-1],999.)
        self.assertEqual(float(prep.named_transformers_['numeric'].statistics_[0]),2.)

    def test_category_identifiers_are_not_passed_as_continuous_values(self):
        frame = pd.DataFrame({'store_nbr':[10,100], 'family':[2,8], 'lag_16':[1.,2.]})
        prep = make_preprocessor(frame.columns)
        result = prep.fit_transform(frame.astype('float32'))
        self.assertEqual(result.dtype,np.float32)
        self.assertEqual(result.shape,(2,5))
        np.testing.assert_array_equal(result[:,:4].sum(axis=1),[2,2])
