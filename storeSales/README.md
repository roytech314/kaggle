# Store Sales - Time Series Forecasting

[Kaggle Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)の売上予測。VS CodeからPythonスクリプトをローカルで実行します。

## 構成

```text
storeSales/
├── baseline.py       # 曜日平均による比較基準・提出CSV出力
├── evaluate.py       # LightGBMの比較評価
├── features.py       # ラグ・移動平均・日時・プロモーション
├── lightgbm_model.py # LightGBMの学習・予測
├── evaluation.py     # 日付分割・RMSLE
├── experiment_history.py # 評価履歴の保存
├── experiments/      # Git管理するスコア・設定・比較表
├── tests/            # 時系列特徴量の検証
├── pyproject.toml    # 依存関係・対応Python
├── uv.lock           # 依存バージョンの固定
├── .python-version   # Python系列
├── data/             # 元CSV
├── result/           # 提出用・確認用CSV
└── docs/architecture.md  # モジュール分割案
```

## 準備と実行

Python 3.14系列と[uv](https://docs.astral.sh/uv/getting-started/installation/)を使用します。

1. コンペに参加し、[Dataページ](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)からデータをダウンロードします。
2. ZIPを展開し、`train.csv`、`test.csv`、`sample_submission.csv` を `storeSales/data/` に置きます。
3. リポジトリ直下から以下を実行します。

```sh
cd storeSales
uv sync --locked
uv run --locked python baseline.py
```

VS CodeではMicrosoftのPython拡張を入れ、「Python: Select Interpreter」で `storeSales/.venv/bin/python` を選択します。

実行結果:

- ターミナル: データ件数・期間・検証RMSLE
- `result/submission.csv`: 提出用予測（`id,sales`）
- `result/train_preview.csv`: 学習データ先頭1,000行

同名の出力CSVは上書きします。提出方法は[コンペの公式手順](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview)を参照してください。

## 予測と検証

店舗・商品カテゴリ・曜日ごとの直近28日平均で予測します。履歴がない組み合わせは、店舗・カテゴリ平均、カテゴリ平均、全体平均の順に補います。

テスト期間と同じ日数を学習データ末尾から取り置き、過去のデータだけで予測してRMSLEを計算します。RMSLEは小さいほど良い指標です。提出用の予測は全学習データから作ります。

## データの確認

Googleスプレッドシートの「ファイル → インポート → アップロード」で、`stores.csv` や `result/train_preview.csv` を読み込みます。先頭1,000行は列や値の形式の確認用です。全体の傾向は日付別・店舗別などの集計で確認します。

元CSVは変更せず、加工処理はPythonコードに残します。スプレッドシートは確認用コピーとして扱います。

## 依存関係の更新

以下は `storeSales/` 内で実行します。

```sh
# 依存関係の確認
uv tree --locked

# パッケージの追加例
uv add scikit-learn

# pandasを宣言済みの範囲内で更新
uv lock --upgrade-package pandas
uv sync --locked

# 更新後の動作確認
uv run --locked python baseline.py
```

`uv.lock` はuvで生成します。更新時は依存ファイルの差分と実行結果を確認してください。Python系列を変更する場合は `.python-version` と `pyproject.toml` の `requires-python` を合わせて更新します。

## Git管理

コード、文書、`pyproject.toml`、`uv.lock`、`.python-version` を管理します。元CSV、出力CSV、`.venv/` は管理対象外です。

## LightGBMの比較評価

macOSではHomebrewのOpenMPライブラリが必要です。初回のみ `brew install libomp` を実行してください。

`storeSales/` 内で実行します。

```sh
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv run --locked python evaluate.py
```

2017-07-31〜2017-08-15の16日間を検証用に固定し、曜日平均と以下の3条件を比較します。LightGBMの学習対象は2016-01-01以降、特徴量の履歴はそれ以前も含みます。

1. 店舗・カテゴリ・プロモーション数・日時
2. 上記＋16・21・28・35・56日前の売上
3. 上記＋16日ずらした7日・28日移動平均

移動平均は予測対象日の16日前を末尾とする窓です。予測期間中の実売上を使わず16日を一括予測します。欠測日を0で埋めず、暦日で計算します。店舗・カテゴリをカテゴリ変数として扱い、目的変数は `log1p(sales)`、学習は固定300回です。検証期間でのearly stoppingやパラメーター探索は行いません。

出力は `result/lightgbm_<実行日時>/` に保存します。

- `metrics.json`: 各方式のRMSLE、期間、設定、環境、入力・コードのハッシュ
- `validation_predictions.csv`: 各検証行の正解・予測
- `*_importance.csv`: 特徴量の分割利得（因果効果を示すものではありません）
- `*.txt`: 学習済みモデル

この評価ではKaggle提出用の再学習・テスト予測は行いません。単一期間の比較なので、改善の安定性は別の期間でも確認してください。

参考: [LightGBMのパラメーター](https://lightgbm.readthedocs.io/en/stable/Parameters.html)、[インストール](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html)。

## 評価履歴のバージョン管理

`evaluate.py` は、実行ごとの評価記録を `experiments/<run_id>.json` に保存し、全記録から `experiments/summary.csv` を再生成します。両方をコードと一緒にコミットしてください。

- JSON: RMSLE、特徴量、モデル設定、学習・検証期間、ライブラリ版、入力・コードのハッシュ
- CSV: 各方式のRMSLEと、同じ実行の曜日平均に対する改善量・改善率

改善量は `baseline_score - score`、改善率はその値を `baseline_score` で割った百分率です。正の値が改善を表します。異なる実行の比較では、入力ハッシュ・学習期間・検証期間・設定を照合してください。過去のJSONを上書きせず、新しい実行として記録します。

指標は売上のRMSEではなく **RMSLE** です。LightGBMの内部設定 `metric=rmse` はlog1p売上に対する指標で、保存するスコアは非負に戻した予測から計算したRMSLEです。

モデル本体と行単位の予測は引き続き `result/` に保存し、Git管理から除外します。
