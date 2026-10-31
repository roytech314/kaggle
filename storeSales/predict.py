"""最新評価で最良だったLightGBM構成を全履歴で再学習し、提出CSVを作る。"""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from features import HORIZON, build_features
from lightgbm_model import PARAMS, ROUNDS, fit_predict

ROOT = Path(__file__).resolve().parent


def main():
    records = sorted((ROOT / 'experiments').glob('*.json'))
    record_path = records[-1]
    evaluation_number = len(records)
    record = json.loads(record_path.read_text())
    best = min(record['scores'], key=record['scores'].get)
    if best not in record['features']:
        raise ValueError('最新評価の最良方式はLightGBMではありません')
    # 評価後にモデルや特徴量が変わっていたら、先に再評価を求める。
    for name in ['features.py', 'lightgbm_model.py']:
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != record['code_sha256'][name]:
            raise ValueError(f'評価時から変更されています: {name}')
    if PARAMS != record['parameters'] or ROUNDS != record['rounds']:
        raise ValueError('評価時の学習設定と一致しません')
    source = ROOT / 'data/train.csv'
    if hashlib.sha256(source.read_bytes()).hexdigest() != record['input_sha256']:
        raise ValueError('学習CSVが評価時と一致しません')
    history = pd.read_csv(source, parse_dates=['date'])
    test = pd.read_csv(ROOT / 'data/test.csv', parse_dates=['date'])
    sample = pd.read_csv(ROOT / 'data/sample_submission.csv')
    if not (history.date.max() < test.date.min() and
            test.date.max() <= history.date.max() + pd.DateOffset(days=HORIZON)):
        raise ValueError('テスト期間は履歴直後の16日以内である必要があります')
    train = history.loc[history.date >= record['train_start']]
    columns = record['features'][best]
    print(f'再学習: {best}, {len(train):,}行, 履歴末尾={history.date.max().date()}', flush=True)
    train_x = build_features(history, train.drop(columns='sales'))[columns]
    test_x = build_features(history, test)[columns]
    pred, model = fit_predict(train_x, train.sales.to_numpy(), test_x)
    assert test.id.is_unique and sample.id.is_unique and set(test.id) == set(sample.id)
    submission = sample[['id']].copy()
    submission['sales'] = submission.id.map(pd.Series(pred, index=test.id))
    assert len(submission) == len(test)
    assert np.isfinite(submission.sales).all() and (submission.sales >= 0).all()
    output = ROOT / 'result'
    output.mkdir(exist_ok=True)
    # eval01は評価記録の1回目。学習方式と日時も名前に含めて識別する。
    topic = best.removeprefix('calendar_promotion_')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    tag = f'lightgbm_{topic}_eval{evaluation_number:02d}_{stamp}'
    submission_path = output / f'submission_{tag}.csv'
    submission.to_csv(submission_path, index=False)
    model.save_model(str(output / f'model_{tag}.txt'))
    metadata = {
        'evaluation_record': record_path.name, 'evaluation_number': evaluation_number,
        'submission_file': submission_path.name, 'model': best,
        'validation_rmsle': record['scores'][best], 'train_start': record['train_start'],
        'train_end': str(history.date.max().date()), 'train_rows': len(train),
        'rows': len(submission), 'features': columns, 'parameters': PARAMS, 'rounds': ROUNDS,
        'input_sha256': {name: hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()
                         for name in ['train.csv', 'test.csv', 'sample_submission.csv']},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'submission_sha256': hashlib.sha256(submission_path.read_bytes()).hexdigest(),
    }
    (output / f'metadata_{tag}.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print('提出ファイル:', submission_path, flush=True)


if __name__ == '__main__':
    main()
