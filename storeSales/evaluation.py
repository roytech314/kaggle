"""モデル間で共通に使う時系列分割と評価指標。"""

from datetime import timedelta
import numpy as np


def split_last_days(data, days=16):
    """末尾days日を検証用にし、同じ日を学習/検証にまたがせない。"""
    start = data.date.max().to_pydatetime() - timedelta(days=days - 1)
    return data.loc[data.date < start].copy(), data.loc[data.date >= start].copy()


def rmsle(actual, predicted):
    if not np.isfinite(predicted).all() or (predicted < 0).any():
        raise ValueError('予測には有限の非負値が必要です')
    return float(np.sqrt(np.mean((np.log1p(actual) - np.log1p(predicted)) ** 2)))
