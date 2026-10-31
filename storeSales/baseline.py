"""店舗・商品カテゴリ・曜日別の直近平均を使う売上予測のベースライン。

処理の流れ:
    CSV読込 → 過去/未来に分けて検証 → 全履歴から予測 → CSV保存

この方式は機械学習モデルのパラメーターを最適化せず、過去の売上平均を使う。
現在はプロモーション、祝日、原油価格、取引件数を予測に使っていない。
予測方法はpredict_sales、検証条件と出力はmainで定義する。
"""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ファイル自身の場所を基準にし、VS Codeの起動フォルダーに依存させない。
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'result'


def predict_sales(history, future, window=28):
    """利用可能な売上履歴から、指定された未来の各行の売上を予測する。

    Args:
        history: date・store_nbr・family・sales列を持つ過去データ。
        future: date・store_nbr・family列を持つ予測対象。sales列は参照しない。
        window: 履歴の最終日から何日分を平均するか。既定は28日。

    Returns:
        futureの行順に対応する、非負の売上予測のNumPy配列。

    入力のDataFrameは変更しない。空データや学習/予測期間の重複はエラーにする。
    検証用と提出用の両方でこの関数を使い、予測方法を一致させる。
    """
    # 1. 予測時点より後の実売上を集計へ混ぜないため、期間を確認する。
    if history.empty or future.empty:
        raise ValueError('学習データと予測対象が必要です')
    if history.date.max() >= future.date.min():
        raise ValueError('学習期間は予測期間より前にしてください')
    # 2. 直近window日分だけを使用する。28日なら通常は各曜日が4回含まれる。
    #    最近の傾向は反映するが、長期の季節性や成長傾向は直接表現しない。
    #    改善の着眼点: 同じ検証期間でwindowを変え、スコアを比較する。
    recent = history.loc[
        history.date > history.date.max().to_pydatetime() - timedelta(days=window)
    ].copy()
    # 月曜=0〜日曜=6。同じ曜日の売れ方を平均で表す。
    recent['weekday'] = recent.date.dt.dayofweek
    # 3. 予測対象にも同じ曜日列を作る。売上の正解はここへ持ち込まない。
    #    結合処理の後も入力と同じ順番に戻せるよう、行番号を保存する。
    target = future[['store_nbr', 'family', 'date']].copy()
    target['weekday'] = target.date.dt.dayofweek
    target['_order'] = np.arange(len(target))
    # 4. 店舗×カテゴリ×曜日ごとの平均を、該当する未来の各行へ結合する。
    #    日付ごとに履歴を更新せず、予測期間全体を同じ過去の集計から予測する。
    #    validateは、平均側のキー重複で予測対象の行が増えることを防ぐ。
    keys = ['store_nbr', 'family', 'weekday']
    means = recent.groupby(keys, observed=True).sales.mean().rename('prediction')
    target = target.merge(means, on=keys, how='left', validate='many_to_one')
    # 5. 該当する曜日の履歴がない場合、より広い単位の平均で補う。
    #    店舗×カテゴリ → カテゴリ → 直近履歴全体、の順で使用する。
    #    平均が0の組み合わせは欠損ではないので、0のまま予測する。
    fallback = recent.groupby(['store_nbr', 'family'], observed=True).sales.mean().rename('fallback')
    target = target.merge(fallback, on=['store_nbr', 'family'], how='left', validate='many_to_one')
    target['prediction'] = target.prediction.fillna(target.fallback)
    family_mean = recent.groupby('family', observed=True).sales.mean()
    target['prediction'] = target.prediction.fillna(target.family.map(family_mean))
    target['prediction'] = target.prediction.fillna(recent.sales.mean())
    # 6. 入力順に戻し、売上として扱える非負の数値配列で返す。
    return target.sort_values('_order').prediction.clip(lower=0).to_numpy()


def main():
    """データ読込・期間分割・精度検証・提出用予測・保存を順に実行する。"""
    # 1. 元CSVを読み込む。dateを日付型にし、期間比較や曜日抽出に使う。
    #    sample_submissionは正解ではなく、提出するIDと列形式の見本。
    OUTPUT_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(DATA_DIR / 'train.csv', parse_dates=['date'])
    test = pd.read_csv(DATA_DIR / 'test.csv', parse_dates=['date'])
    sample = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    print('train:', train.shape, 'test:', test.shape)
    print('学習期間:', train.date.min(), '〜', train.date.max())
    print('予測期間:', test.date.min(), '〜', test.date.max())

    # 2. テスト期間と同じ長さを、学習データの末尾から検証用に取り置く。
    #    現在は16日。日付で分けるため同じ日の全店舗が同じ側に入る。
    #    fitには過去、validにはその後の未来が入り、ランダム分割はしない。
    horizon = (test.date.max() - test.date.min()).days + 1
    valid_start = train.date.max().to_pydatetime() - timedelta(days=horizon - 1)
    fit = train.loc[train.date < valid_start]
    valid = train.loc[train.date >= valid_start]
    # 3. 検証期間の売上を使わず予測し、その後で正解と比較する。
    #    validにsales列は残るが、predict_salesはその列を参照しない。
    valid_pred = predict_sales(fit, valid)
    # RMSLE = log(1 + 売上)の誤差の二乗平均平方根。小さいほど良い。
    # log1pはlog(1+x)なので、売上0も評価できる。
    # 売上そのもののRMSEとは異なり、大きい売上だけが支配しにくくなる。
    # このスコアは過去期間の検証値であり、Kaggle提出スコアとは別。
    rmsle = np.sqrt(np.mean((np.log1p(valid.sales.to_numpy()) - np.log1p(valid_pred)) ** 2))
    print('検証期間:', valid_start.date(), '〜', train.date.max().date())
    print(f'Validation RMSLE: {rmsle:.5f}')

    # 改善の着眼点: 検証開始日をずらした複数期間でも効果が続くか確認する。
    # 方式ごとの検証期間は揃える。特徴量追加では予測時点での入手可否を確認する。

    # 4. 検証が終わったら、検証用に取り置いた期間も含む全履歴を使って予測する。
    #    これにより、テスト開始直前までの情報を提出予測へ反映できる。
    predictions = predict_sales(train, test)
    # 5. IDをキーに提出見本の順序へ並べる。単に行番号で連結すると、
    #    途中のソートや結合で予測が別のIDに割り当たる危険がある。
    assert test.id.is_unique and sample.id.is_unique
    assert set(test.id) == set(sample.id)
    by_id = pd.Series(predictions, index=test.id)
    submission = sample[['id']].copy()
    submission['sales'] = submission.id.map(by_id)
    # 出力前に件数・欠損/無限大・負値を確認する。採点対象との対応を守る確認。
    assert len(submission) == len(test)
    assert np.isfinite(submission.sales).all()
    assert (submission.sales >= 0).all()
    submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
    print('保存先:', OUTPUT_DIR / 'submission.csv')

    # 6. スプレッドシートで列や値を確認するための抜粋。全体の代表標本ではない。
    train.head(1000).to_csv(OUTPUT_DIR / 'train_preview.csv', index=False)
    print('確認用CSV:', OUTPUT_DIR / 'train_preview.csv')


# 直接実行した場合だけ処理を開始する。importでの学習・ファイル保存を防ぐ。
if __name__ == '__main__':
    main()
