"""年代・季節を先に固定した16日間の評価と、学習期間の長さの比較。"""

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline import predict_sales
from evaluation import rmsle
from features import HORIZON, build_features
from lightgbm_model import PARAMS, ROUNDS, fit_predict

ROOT = Path(__file__).resolve().parent
# スコアを見る前に期間を固定。年末区間はクリスマスの欠測日も含む。
STARTS = ('2013-08-01', '2013-12-16', '2014-08-01', '2014-12-16',
          '2016-08-01', '2016-12-16', '2017-04-01', '2017-07-31')
POLICIES = ('era_expanding', 'recent_365_days')


def select_period(data, start, policy):
    """検証は暦日で16日。2013/2016から始まる年代内で過去だけを使う。"""
    start = pd.Timestamp(start)
    era = pd.Timestamp('2013-01-01' if start.year < 2015 else '2016-01-01')
    history = data.loc[(data.date >= era) & (data.date < start)].copy()
    valid = data.loc[(data.date >= start) & (data.date < start + pd.DateOffset(days=HORIZON))].copy()
    if policy not in POLICIES:
        raise ValueError('未知の学習期間です')
    train_start = era if policy == 'era_expanding' else max(era, start - pd.DateOffset(days=365))
    # 初期のラグ欠損が学習量の差を作らないよう、最初の56日間を準備期間にする。
    train_start = max(train_start, era + pd.DateOffset(days=56))
    train = history.loc[history.date >= train_start].copy()
    if train.empty or valid.empty or valid.date.max() != start + pd.DateOffset(days=15):
        raise ValueError('学習または検証期間が不足しています')
    return history, train, valid


def plot_results(rows, data, tag):
    results = pd.DataFrame(rows)
    fig, ax = plt.subplots(2, 1, figsize=(13, 9))
    pivot = results.pivot(index='validation_start', columns='model', values='rmsle')
    pivot.plot.bar(ax=ax[0], color=['#777777', '#2470af', '#d68a26'])
    ax[0].set(title='Fixed 16-calendar-day backtests | lower RMSLE is better', xlabel='', ylabel='RMSLE')
    ax[0].tick_params(axis='x', rotation=30)
    improvement = pivot.sub(pivot['baseline'], axis=0).drop(columns='baseline') * -1
    improvement.plot.bar(ax=ax[1], color=['#2470af', '#d68a26'])
    ax[1].axhline(0, color='black', lw=0.8)
    ax[1].set(title='Improvement vs weekday mean | positive = better', ylabel='Baseline RMSLE - model RMSLE', xlabel='Validation start')
    ax[1].tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(ROOT/'result'/f'backtest_scores_{tag}.png', dpi=150)
    plt.close(fig)
    daily = data.groupby('date').sales.sum()
    monthly = daily.resample('MS').mean()
    table = monthly.to_frame('mean_daily_sales')
    table['year'] = table.index.year
    table['month'] = table.index.month
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for year in [2013, 2014, 2016, 2017]:
        part = table.loc[table.year == year]
        ax[0].plot(part.month, part.mean_daily_sales, marker='o', label=str(year))
        # 年内平均で割り、年をまたぐ規模差と年内パターンを分けて観察する。
        # 2017年は8月途中までなので基準自体も他年と異なる。
        ax[1].plot(part.month, part.mean_daily_sales / part.mean_daily_sales.mean(), marker='o', label=str(year))
    ax[0].set(title='Monthly mean daily sales | absolute', xlabel='Month', ylabel='Sales units / observed day')
    ax[1].set(title='Within-year pattern | each year normalized to its observed mean', xlabel='Month', ylabel='Ratio to observed yearly mean')
    for a in ax:
        a.set_xticks(range(1,13));a.legend();a.grid(alpha=0.2)
    fig.suptitle('2015 excluded | 2017 ends Aug 15 | missing dates are not filled with zero')
    fig.tight_layout()
    fig.savefig(ROOT/'result'/f'backtest_seasonality_{tag}.png', dpi=150)
    plt.close(fig)
    table.to_csv(ROOT/'result'/f'backtest_monthly_{tag}.csv', index_label='month_start')
    return pivot


def main():
    source = ROOT/'data/train.csv'
    data = pd.read_csv(source, parse_dates=['date'])
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    rows = []
    predictions = []
    for start in STARTS:
        for policy in POLICIES:
            history, train, valid = select_period(data, start, policy)
            context = {'validation_start': start, 'validation_end': str((pd.Timestamp(start)+pd.DateOffset(days=15)).date()),
                       'observed_days': valid.date.nunique(), 'validation_rows': len(valid),
                       'train_start': str(train.date.min().date()), 'train_end': str(train.date.max().date()),
                       'train_rows': len(train), 'era': '2013-2014' if pd.Timestamp(start).year < 2015 else '2016-2017'}
            if policy == POLICIES[0]:
                baseline = predict_sales(history, valid.drop(columns='sales'))
                score = rmsle(valid.sales.to_numpy(), baseline)
                rows.append(dict(context, model='baseline', rmsle=score))
                print(f'{start} baseline: {score:.6f}', flush=True)
            # 初期は両ポリシーが同じ学習データになる。その場合は再学習せず同じ予測を使う。
            if policy == POLICIES[0] or not train.id.equals(previous_train_ids):
                train_x = build_features(history, train.drop(columns='sales'))
                valid_x = build_features(history, valid.drop(columns='sales'))
                prediction, _ = fit_predict(train_x, train.sales.to_numpy(), valid_x)
            previous_train_ids = train.id.copy()
            score = rmsle(valid.sales.to_numpy(), prediction)
            rows.append(dict(context, model=policy, rmsle=score))
            detail = valid[['id', 'date', 'sales']].copy()
            detail['model'] = policy;detail['prediction'] = prediction
            if policy == POLICIES[0]: detail['baseline'] = baseline
            predictions.append(detail)
            print(f'{start} {policy}: {score:.6f} ({len(train):,} train rows)', flush=True)
        # 中断時にも完了分の比較結果を残す。
        pd.DataFrame(rows).to_csv(ROOT/'result'/f'backtest_scores_{tag}.csv', index=False)
    pivot = plot_results(rows, data, tag)
    pd.concat(predictions, ignore_index=True).to_csv(ROOT/'result'/f'backtest_predictions_{tag}.csv', index=False)
    summary = []
    for era, dates in [('2013-2014', [s for s in STARTS if s < '2015']),
                       ('2016-2017', [s for s in STARTS if s > '2015'])]:
        for name in pivot.columns:
            values = pivot.loc[dates, name]
            summary.append({'era': era, 'model': name, 'mean_rmsle': float(values.mean()),
                            'min_rmsle': float(values.min()), 'max_rmsle': float(values.max()),
                            'wins_vs_baseline': int((values < pivot.loc[dates, 'baseline']).sum())})
    report = {'run_id': f'backtest_{tag}', 'metric': 'RMSLE', 'starts_fixed_before_evaluation': STARTS,
              'horizon_calendar_days': HORIZON, 'warmup_days': 56, 'rounds': ROUNDS, 'parameters': PARAMS,
              'rows': rows, 'summary': summary, 'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'python': platform.python_version(),
              'libraries': {n: version(n) for n in ['numpy','pandas','lightgbm']},
              'code_sha256': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                              ['backtest.py','features.py','lightgbm_model.py','baseline.py','evaluation.py','uv.lock']}}
    # 単一期間の評価記録とは別に管理し、提出スクリプトの自動選択に混ぜない。
    folder = ROOT/'experiments/backtests';folder.mkdir(exist_ok=True)
    (folder/f'{tag}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print('run:', tag, flush=True)


if __name__ == '__main__':
    main()
