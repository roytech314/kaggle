"""log1p売上の二乗誤差を学習し、RMSLEに対応する売上予測を返す。"""

import lightgbm as lgb
import numpy as np

PARAMS = {
    'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.05,
    'num_leaves': 31, 'min_data_in_leaf': 100, 'verbosity': -1,
    'num_threads': 4, 'seed': 42, 'deterministic': True, 'force_col_wise': True,
}
ROUNDS = 300


def fit_predict(train_x, sales, future_x):
    """固定300回で学習する。評価期間をearly stoppingや調整には使わない。"""
    dataset = lgb.Dataset(train_x, label=np.log1p(sales),
                          categorical_feature=['store_nbr', 'family'])
    model = lgb.train(PARAMS, dataset, num_boost_round=ROUNDS)
    prediction = np.expm1(model.predict(future_x)).clip(min=0)
    return prediction, model
