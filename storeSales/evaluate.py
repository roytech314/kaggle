"""ベースラインとLightGBMを同じ未使用の16日間で比較する実行入口。"""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
from importlib.metadata import version

import pandas as pd

from baseline import predict_sales
from evaluation import rmsle, split_last_days
from experiment_history import record_experiment
from features import HORIZON, LAGS, WINDOWS, build_features
from lightgbm_model import PARAMS, ROUNDS, fit_predict

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / 'data/train.csv'
    data = pd.read_csv(source, parse_dates=['date'])
    history, valid = split_last_days(data, HORIZON)
    # 学習対象を2016年以降に限定。特徴量用の過去履歴はそれ以前も保持する。
    train = history.loc[history.date >= '2016-01-01'].copy()
    # 正解列を除いて渡し、特徴量生成から評価期間の売上を隔離する。
    print('特徴量を生成中...', flush=True)
    train_x = build_features(history, train.drop(columns='sales'))
    valid_x = build_features(history, valid.drop(columns='sales'))
    output = ROOT / 'result' / ('lightgbm_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True)
    predictions = valid[['id', 'date', 'store_nbr', 'family', 'sales']].copy()
    scores = {}
    basic = [name for name in train_x if not name.startswith(('lag_', 'rolling_'))]
    experiments = {
        'calendar_promotion': basic,
        'calendar_promotion_lags': basic + [name for name in train_x if name.startswith('lag_')],
        'calendar_promotion_lags_rolling': list(train_x.columns),
    }
    baseline = predict_sales(history, valid.drop(columns='sales'))
    predictions['baseline'] = baseline
    scores['baseline'] = rmsle(valid.sales.to_numpy(), baseline)
    print(f"baseline RMSLE: {scores['baseline']:.6f}", flush=True)
    for name, columns in experiments.items():
        print(f'学習中: {name} ({len(train):,}行)', flush=True)
        pred, model = fit_predict(train_x[columns], train.sales.to_numpy(), valid_x[columns])
        predictions[name] = pred
        scores[name] = rmsle(valid.sales.to_numpy(), pred)
        model.save_model(str(output / f'{name}.txt'))
        pd.DataFrame({'feature': columns, 'gain': model.feature_importance('gain')}).sort_values(
            'gain', ascending=False).to_csv(output / f'{name}_importance.csv', index=False)
        print(f'{name} RMSLE: {scores[name]:.6f}', flush=True)
    predictions.to_csv(output / 'validation_predictions.csv', index=False)
    report = {
        'run_id': output.name, 'metric': 'RMSLE',
        'baseline_parameters': {'window': 28},
        'scores': scores, 'validation_start': str(valid.date.min().date()),
        'validation_end': str(valid.date.max().date()), 'validation_rows': len(valid),
        'train_start': str(train.date.min().date()), 'train_end': str(train.date.max().date()),
        'train_rows': len(train), 'horizon': HORIZON, 'lags': LAGS, 'rolling_windows': WINDOWS,
        'parameters': PARAMS, 'rounds': ROUNDS, 'features': experiments,
        'python': platform.python_version(),
        'libraries': {name: version(name) for name in ['numpy', 'pandas', 'lightgbm']},
        'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [Path(__file__), ROOT/'features.py', ROOT/'evaluation.py',
                                  ROOT/'lightgbm_model.py', ROOT/'baseline.py', ROOT/'uv.lock',
                                  ROOT/'experiment_history.py', ROOT/'pyproject.toml',
                                  ROOT/'.python-version']},
    }
    (output / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    record = record_experiment(report, ROOT / 'experiments')
    print('保存先:', output, flush=True)
    print('Git管理する評価記録:', record, flush=True)


if __name__ == '__main__':
    main()
