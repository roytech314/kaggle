"""16日先まで一括予測するための、未来の実売上を使わない特徴量。"""

import numpy as np
import pandas as pd

HORIZON = 16
LAGS = (16, 21, 28, 35, 56)
WINDOWS = (7, 28)


def build_features(history, targets):
    """historyの売上とtargetsの既知入力から、targetsの行順で特徴量を返す。

    t日の特徴量はt-16日以前の売上だけを参照する。移動平均も16日シフト
    してから計算するため、予測期間内の正解を翌日の特徴量に使わない。
    欠測日は0と仮定せずNaNで保持し、LightGBMに処理させる。
    """
    if history.duplicated(['date', 'store_nbr', 'family']).any():
        raise ValueError('店舗・カテゴリ・日付は一意である必要があります')
    families = sorted(history.family.unique())
    family_codes = targets.family.map({value: i for i, value in enumerate(families)})
    if family_codes.isna().any():
        raise ValueError('学習履歴にないカテゴリです')
    dates = targets.date.dt
    x = pd.DataFrame({
        'store_nbr': targets.store_nbr.to_numpy(),
        'family': family_codes.to_numpy(),
        'onpromotion': targets.onpromotion.to_numpy(),
        'weekday': dates.dayofweek.to_numpy(),
        'day': dates.day.to_numpy(),
        'month': dates.month.to_numpy(),
        'year': dates.year.to_numpy(),
        'dayofyear': dates.dayofyear.to_numpy(),
    })
    panel = history.pivot(index='date', columns=['store_nbr', 'family'], values='sales')
    # 行数のshiftでは欠測日を飛ばしてしまうため、先に暦日で連続化する。
    end = max(history.date.max(), targets.date.max())
    panel = panel.reindex(pd.date_range(history.date.min(), end, freq='D'))
    row = panel.index.get_indexer(targets.date)
    col = panel.columns.get_indexer(pd.MultiIndex.from_frame(targets[['store_nbr', 'family']]))
    if (row < 0).any() or (col < 0).any():
        raise ValueError('予測対象の日時・系列が履歴に対応していません')
    for lag in LAGS:
        x[f'lag_{lag}'] = panel.shift(lag).to_numpy(dtype='float32')[row, col]
    shifted = panel.shift(HORIZON)
    for window in WINDOWS:
        x[f'rolling_mean_{window}_shift_{HORIZON}'] = (
            shifted.rolling(window, min_periods=window).mean()
            .to_numpy(dtype='float32')[row, col]
        )
    return x
