"""直近28日の曜日別平均を比較基準として、検証と提出用CSVの作成を行う。"""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'result'


def predict_sales(history, future, window=28):
    if history.empty or future.empty:
        raise ValueError('学習データと予測対象が必要です')
    if history.date.max() >= future.date.min():
        raise ValueError('学習期間は予測期間より前にしてください')
    recent = history.loc[
        history.date > history.date.max().to_pydatetime() - timedelta(days=window)
    ].copy()
    recent['weekday'] = recent.date.dt.dayofweek
    # merge後も入力と同じ順番に戻せるようにする。
    target = future[['store_nbr', 'family', 'date']].copy()
    target['weekday'] = target.date.dt.dayofweek
    target['_order'] = np.arange(len(target))
    keys = ['store_nbr', 'family', 'weekday']
    means = recent.groupby(keys, observed=True).sales.mean().rename('prediction')
    target = target.merge(means, on=keys, how='left', validate='many_to_one')
    fallback = recent.groupby(['store_nbr', 'family'], observed=True).sales.mean().rename('fallback')
    target = target.merge(fallback, on=['store_nbr', 'family'], how='left', validate='many_to_one')
    target['prediction'] = target.prediction.fillna(target.fallback)
    family_mean = recent.groupby('family', observed=True).sales.mean()
    target['prediction'] = target.prediction.fillna(target.family.map(family_mean))
    target['prediction'] = target.prediction.fillna(recent.sales.mean())
    return target.sort_values('_order').prediction.clip(lower=0).to_numpy()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(DATA_DIR / 'train.csv', parse_dates=['date'])
    test = pd.read_csv(DATA_DIR / 'test.csv', parse_dates=['date'])
    sample = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    print('train:', train.shape, 'test:', test.shape)
    print('学習期間:', train.date.min(), '〜', train.date.max())
    print('予測期間:', test.date.min(), '〜', test.date.max())

    horizon = (test.date.max() - test.date.min()).days + 1
    valid_start = train.date.max().to_pydatetime() - timedelta(days=horizon - 1)
    fit = train.loc[train.date < valid_start]
    valid = train.loc[train.date >= valid_start]
    valid_pred = predict_sales(fit, valid)
    rmsle = np.sqrt(np.mean((np.log1p(valid.sales.to_numpy()) - np.log1p(valid_pred)) ** 2))
    print('検証期間:', valid_start.date(), '〜', train.date.max().date())
    print(f'Validation RMSLE: {rmsle:.5f}')

    predictions = predict_sales(train, test)
    assert test.id.is_unique and sample.id.is_unique
    assert set(test.id) == set(sample.id)
    by_id = pd.Series(predictions, index=test.id)
    submission = sample[['id']].copy()
    submission['sales'] = submission.id.map(by_id)
    assert len(submission) == len(test)
    assert np.isfinite(submission.sales).all()
    assert (submission.sales >= 0).all()
    submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
    print('保存先:', OUTPUT_DIR / 'submission.csv')

    train.head(1000).to_csv(OUTPUT_DIR / 'train_preview.csv', index=False)
    print('確認用CSV:', OUTPUT_DIR / 'train_preview.csv')


if __name__ == '__main__':
    main()
