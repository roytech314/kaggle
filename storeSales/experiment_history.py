"""評価の設定・スコアをGit管理する小さなJSONと比較表に保存する。"""

import csv
import json
import math
from pathlib import Path

FIELDS = ['run_id', 'metric', 'validation_start', 'validation_end', 'model',
          'score', 'baseline_score', 'improvement', 'improvement_pct', 'input_sha256']


def record_experiment(report, directory):
    """実行ごとの記録を追加し、履歴全体からsummary.csvを再生成する。

    improvementは「同じ実行のベースライン − モデル」のRMSLE。
    正の値が改善。異なる検証期間や入力データ間の差は算出しない。
    同じrun_idに異なる記録を上書きすることは禁止する。
    """
    directory = Path(directory)
    run_id = report['run_id']
    if Path(run_id).name != run_id or run_id in {'.', '..', ''}:
        raise ValueError('run_idはファイル名として使える識別子にしてください')
    if report['metric'] != 'RMSLE':
        raise ValueError('記録対象の指標はRMSLEです')
    scores = report['scores']
    if 'baseline' not in scores or any(not math.isfinite(s) or s < 0 for s in scores.values()):
        raise ValueError('有限で非負のスコアとbaselineが必要です')
    content = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f'{run_id}.json'
    if destination.exists() and destination.read_text() != content:
        raise ValueError(f'既存の実験記録を上書きできません: {run_id}')
    destination.write_text(content)
    rows = []
    for path in sorted(directory.glob('*.json')):
        item = json.loads(path.read_text())
        baseline = item['scores']['baseline']
        for model, score in item['scores'].items():
            delta = baseline - score
            rows.append({
                'run_id': item['run_id'], 'metric': item['metric'],
                'validation_start': item['validation_start'],
                'validation_end': item['validation_end'], 'model': model,
                'score': score, 'baseline_score': baseline, 'improvement': delta,
                'improvement_pct': delta / baseline * 100 if baseline else '',
                'input_sha256': item['input_sha256'],
            })
    with (directory / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return destination
