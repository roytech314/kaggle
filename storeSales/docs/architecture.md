# モジュール分割案

現在の実行入口は `baseline.py`。モデル比較や特徴量追加に向け、処理を以下の4モジュールへ分割する。

| モジュール | 責任 |
|---|---|
| `run.py` | 設定と実行順を管理する唯一の入口 |
| `data_io.py` | CSV読込、必須列の検証、提出ID順の整列、出力検証・保存 |
| `baseline.py` | 履歴と予測対象を受け取り、入力順に予測値を返す |
| `evaluation.py` | 日付で学習・検証を分割し、RMSLEを計算する |

`data/`・`result/` は他コンペと揃えて維持する。各モジュールは関数を通じてデータを受け渡し、import時には読込・学習・保存を実行しない。`run.py` が各モジュールを呼び、逆方向の依存は作らない。

## データの流れ

1. CSVを読み込み、列・日付・IDを検証する。
2. 末尾16日を検証用に分け、学習部分から予測して評価する。
3. 全学習データからテスト期間を予測し、提出用CSVを保存する。
4. 閲覧用の抽出・集計結果を確認用CSVに保存する。

## 時系列の制約

- 同じ日付の全店舗・商品カテゴリを同じ側に分割する。
- 予測起点より後の実売上を特徴量に含めない。16日先まで予測する場合、検証期間内の真の売上をラグや移動平均に使わない。
- 学習を伴う前処理は学習部分だけでfitし、同じ変換を検証・テストへ適用する。
- 外部変数は予測時点で取得可能か確認する。`transactions.csv` にはテスト期間の実績が含まれない。

## 拡張の目安

探索・集計が増えたら `survey.py`、複数モデルで使う特徴量が増えたら `features.py` を追加する。学習済み状態を再利用する必要が出た段階で学習と予測を分ける。

実験比較では `result/<run_id>/` に予測値、設定、評価期間、RMSLE、入力データとコードの識別情報、実行環境のバージョンを記録する。

## 分割時の確認

同一データ・環境・設定で、変更前後の予測値と丸め前のRMSLEを比較する。提出ID順・件数・有限値・非負、元CSVが変更されないこと、import時に処理が実行されないことを確認する。

## 参考資料

- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/): データ処理・特徴量・学習・予測などの構成例。
- [同設計方針](https://cookiecutter-data-science.drivendata.org/opinions/): 元データの保持と再現可能な処理。
- [Pythonの実行入口](https://docs.python.org/3/library/__main__.html): import時の意図しない実行を防ぐ。
- [scikit-learnの注意点](https://scikit-learn.org/stable/common_pitfalls.html): 前処理の一致とデータ漏洩の防止。
- [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html): 時刻順の検証。日付単位の分割は本データに合わせた設計判断。
