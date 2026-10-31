# Store Sales - Time Series Forecasting

[Kaggle Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)の売上予測。VS CodeからPythonスクリプトをローカルで実行します。

## 構成

```text
storeSales/
├── baseline.py       # 予測・時系列検証・CSV出力
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
