"""カテゴリをone-hot化し、log1p売上をランダムフォレストで学習する。"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

PARAMS = {
    'n_estimators': 100, 'max_depth': 18, 'min_samples_leaf': 20,
    'max_features': 0.8, 'bootstrap': True, 'max_samples': 0.3,
    'n_jobs': 4, 'random_state': 42,
}
CATEGORIES = ['store_nbr', 'family']


def make_preprocessor(columns):
    """学習部分でのみカテゴリ一覧と欠損補完値を確定する。

    店舗番号・カテゴリコードを大小関係として扱わずone-hot化する。
    未知カテゴリは全0、数値の欠測は学習側の中央値で補完する。
    """
    return ColumnTransformer([
        ('categories', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32), CATEGORIES),
        ('numeric', SimpleImputer(strategy='median', keep_empty_features=True),
         [c for c in columns if c not in CATEGORIES]),
    ], sparse_threshold=0)


def fit_predict(train_x, sales, future_x):
    """固定設定で学習。木ごとの30%抽出はRFだけの学習設定で、検証行は全件評価。"""
    model = make_pipeline(make_preprocessor(train_x.columns), RandomForestRegressor(**PARAMS))
    # float32で大きなone-hot行列のメモリを抑える。補完値もtrainだけで学習する。
    model.fit(train_x.astype('float32'), np.log1p(sales))
    prediction = np.expm1(model.predict(future_x.astype('float32'))).clip(min=0)
    return prediction, model
