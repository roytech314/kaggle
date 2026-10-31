"""2016〜2017年の固定4期間で、曜日平均・LightGBM・RFを比較する入口。"""

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import hashlib
import json
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest import select_period
from baseline import predict_sales
from evaluation import rmsle
from features import build_features
from lightgbm_model import PARAMS as LGB_PARAMS, ROUNDS, fit_predict as fit_lgb
from random_forest_model import PARAMS as RF_PARAMS, fit_predict as fit_rf

ROOT = Path(__file__).resolve().parent
STARTS = ('2016-08-01', '2016-12-16', '2017-04-01', '2017-07-31')


def main():
    source = ROOT/'data/train.csv'
    data = pd.read_csv(source, parse_dates=['date'])
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    rows, details = [], []
    features = None
    for start in STARTS:
        history, train, valid = select_period(data, start, 'era_expanding')
        print(f'{start}: {len(train):,}学習行 / {len(valid):,}検証行', flush=True)
        train_x = build_features(history, train.drop(columns='sales'))
        valid_x = build_features(history, valid.drop(columns='sales'))
        features = list(train_x.columns)
        for name, fit in [('baseline', None), ('lightgbm', fit_lgb), ('random_forest', fit_rf)]:
            print(f'学習中: {name}', flush=True)
            begin = perf_counter()
            if fit is None:
                pred = predict_sales(history, valid.drop(columns='sales'))
            else:
                pred, model = fit(train_x, train.sales.to_numpy(), valid_x)
                del model
            elapsed = perf_counter() - begin
            score = rmsle(valid.sales.to_numpy(), pred)
            detail = valid[['id','date','sales']].copy()
            detail['model'] = name; detail['prediction'] = pred
            detail['validation_start'] = start
            details.append(detail)
            rows.append({'validation_start': start, 'validation_end': str(valid.date.max().date()),
                         'train_start': str(train.date.min().date()) if fit else str((history.date.max()-pd.DateOffset(days=27)).date()),
                         'train_end': str(train.date.max().date()),
                         'train_rows': len(train) if fit else len(history.loc[history.date > history.date.max()-pd.DateOffset(days=28)]),
                         'validation_rows': len(valid), 'model': name, 'rmsle': score,
                         'fit_predict_seconds': elapsed})
            print(f'{name}: {score:.6f} ({elapsed:.1f}秒)', flush=True)
            # 中断時も完了した計算結果は残す。
            pd.DataFrame(rows).to_csv(ROOT/'result'/f'model_comparison_{tag}.csv', index=False)
        del train_x, valid_x
    detail = pd.concat(details, ignore_index=True)
    detail.to_csv(ROOT/'result'/f'model_comparison_predictions_{tag}.csv', index=False)
    frame = pd.DataFrame(rows)
    pivot = frame.pivot(index='validation_start',columns='model',values='rmsle')[['baseline','lightgbm','random_forest']]
    summary = []
    for name in pivot:
        part = detail.loc[detail.model == name]
        summary.append({'model': name, 'mean_rmsle': float(pivot[name].mean()),
                        'pooled_rmsle': rmsle(part.sales.to_numpy(),part.prediction.to_numpy()),
                        'std_rmsle': float(pivot[name].std(ddof=0)),
                        'wins_vs_baseline': int((pivot[name]<pivot.baseline).sum()),
                        'total_fit_predict_seconds': float(frame.loc[frame.model==name,'fit_predict_seconds'].sum())})
    fig, ax = plt.subplots(1,2,figsize=(14,5))
    pivot.plot.bar(ax=ax[0],color=['#777777','#2470af','#d68a26'])
    ax[0].set(title='Same 4 periods | lower RMSLE is better',ylabel='RMSLE',xlabel='Validation start')
    ax[0].tick_params(axis='x',rotation=25)
    summary_frame = pd.DataFrame(summary).set_index('model')
    summary_frame.mean_rmsle.plot.bar(ax=ax[1],color=['#777777','#2470af','#d68a26'])
    ax[1].set(title='Mean of 4 period scores',ylabel='Mean RMSLE',xlabel='')
    ax[1].tick_params(axis='x',rotation=15)
    fig.tight_layout();fig.savefig(ROOT/'result'/f'model_comparison_{tag}.png',dpi=150);plt.close(fig)
    report = {'run_id': f'model_comparison_{tag}', 'metric':'RMSLE', 'validation_starts': STARTS,
              'training_policy':'era_expanding', 'warmup_days':56, 'features':features,
              'target':'log1p(sales)', 'baseline_window':28, 'lightgbm_parameters':LGB_PARAMS,
              'lightgbm_rounds':ROUNDS, 'random_forest_parameters':RF_PARAMS,
              'categorical_handling': {'lightgbm':'native', 'random_forest':'one-hot, fit on train only'},
              'numeric_missing_handling': {'lightgbm':'native', 'random_forest':'median fitted on train only'},
              'rows': rows, 'summary':summary, 'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'python':platform.python_version(),
              'libraries':{n:version(n) for n in ['numpy','pandas','lightgbm','scikit-learn']},
              'code_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                             ['compare_models.py','random_forest_model.py','lightgbm_model.py','features.py',
                              'backtest.py','baseline.py','evaluation.py','uv.lock']}}
    folder=ROOT/'experiments/model_comparisons';folder.mkdir(exist_ok=True)
    (folder/f'{tag}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(summary_frame.to_string(),flush=True)
    print('run:',tag,flush=True)


if __name__ == '__main__':
    main()
