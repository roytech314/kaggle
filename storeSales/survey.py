"""学習CSVの分布・時系列・偏り・相関を集計し、result直下へ図と概要を出力する。"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'result'
COLOR = '#2563a6'


def save(fig, name):
    fig.tight_layout(pad=2)
    fig.savefig(OUT / f'eda_{name}.png', dpi=150, facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    train = pd.read_csv(ROOT / 'data/train.csv', parse_dates=['date'])
    test = pd.read_csv(ROOT / 'data/test.csv', parse_dates=['date'])
    stores = pd.read_csv(ROOT / 'data/stores.csv')
    oil = pd.read_csv(ROOT / 'data/oil.csv', parse_dates=['date'])
    transactions = pd.read_csv(ROOT / 'data/transactions.csv', parse_dates=['date'])
    holidays = pd.read_csv(ROOT / 'data/holidays_events.csv', parse_dates=['date'])
    daily = train.groupby('date').agg(sales=('sales', 'sum'), onpromotion=('onpromotion', 'sum'))
    family = train.groupby('family').agg(total=('sales', 'sum'), mean=('sales', 'mean'),
                                        zero_rate=('sales', lambda x: x.eq(0).mean())).sort_values('total', ascending=False)
    missing_dates = pd.date_range(train.date.min(), train.date.max()).difference(daily.index)
    # 元データは変更しない。全体像とモデル学習に近い2016年以降を区別する。
    recent = train.loc[train.date >= '2016-01-01'].copy()
    print('集計・描画中...', flush=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.grid': True, 'grid.alpha': 0.2})

    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    # 欠測日は0として描かず、NaNで途切れさせる。
    daily.sales.reindex(pd.date_range(daily.index.min(), daily.index.max())).plot(ax=ax[0, 0], color=COLOR, lw=0.8)
    ax[0, 0].axvspan(pd.Timestamp('2017-07-31'), pd.Timestamp('2017-08-15'), color='orange', alpha=0.3)
    ax[0, 0].set(title='Daily total sales | orange = local validation', ylabel='Sum of sales units', xlabel='')
    daily.sales.groupby(daily.index.year).mean().plot.bar(ax=ax[0, 1], color=COLOR)
    ax[0, 1].set(title='Mean daily total by year | 2017 is partial', ylabel='Sales units / observed day', xlabel='')
    daily.sales.groupby(daily.index.dayofweek).mean().plot.bar(ax=ax[1, 0], color=COLOR)
    ax[1, 0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
    ax[1, 0].set(title='Mean daily total by weekday | all years', ylabel='Sales units / observed day', xlabel='')
    daily.sales.resample('MS').mean().plot(ax=ax[1, 1], color=COLOR, marker='.', lw=1)
    ax[1, 1].set(title='Monthly mean daily sales | last month is partial', ylabel='Sales units / observed day', xlabel='')
    save(fig, '01_time_patterns')

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    counts = [(train.sales == 0).sum(), (train.sales > 0).sum()]
    ax[0, 0].bar(['Zero', 'Positive'], np.array(counts) / len(train) * 100, color=[COLOR, '#e09338'])
    ax[0, 0].set(title='Sales: zero vs positive rows', ylabel='% of training rows')
    ax[0, 1].hist(np.log1p(train.sales), bins=70, color=COLOR)
    ax[0, 1].set(title='Sales distribution (log1p scale)', xlabel='log(1 + sales)', ylabel='Rows')
    (family.total.head(15).iloc[::-1] / family.total.sum() * 100).plot.barh(ax=ax[1, 0], color=COLOR)
    ax[1, 0].set(title='Top 15 families: share of total sales', xlabel='% of sales units', ylabel='')
    (family.zero_rate.sort_values().tail(15) * 100).plot.barh(ax=ax[1, 1], color=COLOR)
    ax[1, 1].set(title='15 families with highest zero-sales rate', xlabel='% zero rows', ylabel='')
    save(fig, '02_distribution')

    fig, ax = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [3, 1]})
    heat = train.pivot_table(index='store_nbr', columns='family', values='sales', aggfunc='mean')
    im = ax[0].imshow(np.log1p(heat), aspect='auto', cmap='YlGnBu')
    ax[0].set_xticks(range(len(heat.columns)), heat.columns, rotation=90, fontsize=8)
    ax[0].set_yticks(range(0, len(heat), 3), heat.index[::3], fontsize=8)
    ax[0].set(title='Mean sales by store and family | log1p', ylabel='Store number')
    ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], shrink=0.6, label='log(1 + mean sales)')
    store_sum = train.groupby('store_nbr').sales.sum().nlargest(15).iloc[::-1]
    (store_sum / train.sales.sum() * 100).plot.barh(ax=ax[1], color=COLOR)
    ax[1].set(title='Top 15 stores', xlabel='% of sales units', ylabel='Store number')
    save(fig, '03_store_family')

    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    sample = recent.sample(min(40000, len(recent)), random_state=42)
    ax[0, 0].scatter(sample.onpromotion, np.log1p(sample.sales), s=2, alpha=0.08, color=COLOR, rasterized=True)
    ax[0, 0].set(title='Promotion vs sales | 40k sampled rows, 2016+', xlabel='Number of promoted items', ylabel='log(1 + sales)')
    bins = pd.cut(recent.onpromotion, [-1, 0, 5, 20, 50, np.inf], labels=['0', '1-5', '6-20', '21-50', '51+'])
    promotion = recent.groupby(bins, observed=True).agg(mean_sales=('sales', 'mean'), rows=('sales', 'size'))
    promotion.mean_sales.plot.bar(ax=ax[0, 1], color=COLOR)
    ax[0, 1].set(title='Mean sales by promotion count | mixed families', xlabel='Promoted items', ylabel='Mean sales')
    promotion.rows.plot.bar(ax=ax[1, 0], color=COLOR)
    ax[1, 0].set(title='Rows in each promotion bin | 2016+', xlabel='Promoted items', ylabel='Rows')
    train.assign(promoted=train.onpromotion.gt(0)).groupby(train.date.dt.year).promoted.mean().mul(100).plot.bar(ax=ax[1, 1], color=COLOR)
    ax[1, 1].set(title='Share of rows with promotion by year', ylabel='% promoted rows', xlabel='')
    save(fig, '04_promotion')

    # 売上/販促/取引件数は日次に集計し、原油の観測がある日のみで比較する。
    # 原油を補完せず、内側結合+dropnaで比較対象日を明示する。
    external = daily.join(transactions.groupby('date').transactions.sum(), how='inner').join(
        oil.set_index('date')[['dcoilwtico']], how='inner')
    external = external.loc[external.index >= '2016-01-01'].dropna()
    external.columns = ['sales', 'promoted_items', 'transactions', 'oil_price']
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    correlations = {}
    for axis, method in zip(ax, ['pearson', 'spearman']):
        corr = external.corr(method=method)
        correlations[method] = corr.to_dict()
        im = axis.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
        axis.set_xticks(range(4), corr.columns, rotation=30, ha='right')
        axis.set_yticks(range(4), corr.index)
        axis.set_title(f'{method.title()} | daily aggregates, 2016+ | n={len(external)} days')
        axis.grid(False)
        for i in range(4):
            for j in range(4): axis.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', color='white' if abs(corr.iloc[i,j]) > 0.6 else 'black')
    fig.suptitle('Association only: aggregation, trend and family mix can confound correlations', y=1.02)
    fig.subplots_adjust(left=0.12, bottom=0.25, right=0.86, top=0.82, wspace=0.6)
    fig.colorbar(im, cax=fig.add_axes([0.91, 0.25, 0.015, 0.5]))
    fig.savefig(OUT/'eda_05_correlations.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    tables = {'train': train, 'test': test, 'stores': stores, 'oil': oil,
              'transactions': transactions, 'holidays_events': holidays}
    missing = {f'{name}.{col}': float(df[col].isna().mean()*100)
               for name, df in tables.items() for col in df if df[col].isna().any()}
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    ax[0, 0].barh(list(missing) or ['No missing cells'], list(missing.values()) or [0], color=COLOR)
    ax[0, 0].set(title='Missing values in supplied rows', xlabel='% missing')
    # セルの欠損と、行自体がない期間は別に見る。
    monthly_rows = train.groupby(train.date.dt.to_period('M')).size()
    ax[0, 1].plot(monthly_rows.index.to_timestamp(), monthly_rows, color=COLOR)
    ax[0, 1].set(title='Rows per month | partial months are smaller', ylabel='Rows')
    store_counts = stores.groupby('type').size()
    store_counts.plot.bar(ax=ax[1, 0], color=COLOR)
    ax[1, 0].set(title='Store type distribution', ylabel='Stores', xlabel='Type')
    holidays.groupby(['locale','type']).size().unstack(fill_value=0).plot.bar(ax=ax[1, 1], stacked=True)
    ax[1, 1].set(title='Holiday/event records by scope and type', ylabel='Records (not unique dates)', xlabel='')
    ax[1, 1].legend(fontsize=7)
    save(fig, '06_quality_context')

    rowcorr = recent[['sales', 'onpromotion']].corr().iloc[0,1]
    stats = {
        'rows': {name: len(df) for name, df in tables.items()},
        'train_start': str(train.date.min().date()), 'train_end': str(train.date.max().date()),
        'test_start': str(test.date.min().date()), 'test_end': str(test.date.max().date()),
        'test_days': int(test.date.nunique()), 'stores': int(train.store_nbr.nunique()),
        'families': int(train.family.nunique()), 'series': int(train.groupby(['store_nbr','family']).ngroups),
        'zero_sales_pct': float(train.sales.eq(0).mean()*100),
        'sales_quantiles': train.sales.quantile([0,.25,.5,.75,.95,.99,1]).to_dict(),
        'top5_family_sales_share_pct': float(family.total.head(5).sum()/family.total.sum()*100),
        'missing_cell_pct': missing, 'missing_calendar_dates': [str(d.date()) for d in missing_dates],
        'duplicate_series_dates': int(train.duplicated(['date','store_nbr','family']).sum()),
        'row_sales_promotion_pearson_2016plus': float(rowcorr),
        'daily_correlation_days_2016plus': len(external), 'daily_correlations': correlations,
    }
    (OUT/'eda_statistics.json').write_text(json.dumps(stats, indent=2)+'\n')
    family.to_csv(OUT/'eda_family_summary.csv')
    external.to_csv(OUT/'eda_daily_external.csv')
    print(json.dumps({k:v for k,v in stats.items() if k!='daily_correlations'}, indent=2), flush=True)
    print('保存先:', OUT, flush=True)


if __name__ == '__main__':
    main()
