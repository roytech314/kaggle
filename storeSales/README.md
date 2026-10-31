# Store Sales - Time Series Forecasting

[Kaggle Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)の売上予測。VS CodeからPythonスクリプトをローカルで実行します。

データ・検証・提出結果と次の改善候補は[評価結果と改善方針](docs/findings.md)を参照してください。

## 構成

```text
storeSales/
├── baseline.py       # 曜日平均による比較基準・提出CSV出力
├── evaluate.py       # LightGBMの比較評価
├── predict.py        # 最良構成の再学習・提出CSV作成
├── features.py       # ラグ・移動平均・日時・プロモーション
├── lightgbm_model.py # LightGBMの学習・予測
├── evaluation.py     # 日付分割・RMSLE
├── experiment_history.py # 評価履歴の保存
├── experiments/      # Git管理するスコア・設定・比較表
├── survey.py         # データ概要・探索用の画像と集計
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

## LightGBMの提出CSVを作成する

`storeSales/` 内で実行します。

```sh
uv run --locked python predict.py
```

最新の評価記録でRMSLEが最小のLightGBM構成を使い、評価時の学習開始日から元データの最終日までで再学習します。特徴量・学習設定・元CSVが評価時と一致しない場合は停止します。

出力は `result/` 直下の `submission_lightgbm_<方式>_eval<評価回>_<実行日時>.csv` です。例えば `lags_rolling_eval01` は、ラグ・移動平均を使う第1回評価の構成です。モデルとメタデータも同じ識別子を付けた `model_*.txt`・`metadata_*.json` として直下に保存します。メタデータには参照した評価記録・学習期間・入出力ハッシュを記録します。Kaggleへの提出はCSV生成後に手動で行います。評価RMSLEと提出後のリーダーボードスコアは別です。

Kaggle提出後のスコアは `experiments/submissions.csv` に記録します。ローカル検証値とPublicスコアを別列にし、提出CSVの名前・SHA256・評価記録に対応づけます。`score_source=user_reported` はユーザーから報告された値です。

## データの概要と可視化

[コンペとデータのサマリー](docs/data-overview.md)を参照してください。図と集計は `storeSales/` 内で以下を実行すると `result/` 直下に生成できます。

```sh
uv run --locked python survey.py
```

`eda_01_*.png`〜`eda_06_*.png` が画像、`eda_statistics.json` が集計値です。元データは変更しません。同名の可視化出力は上書きします。

## 年代・季節別の16日検証

`storeSales/` 内で実行します。

```sh
uv run --locked python backtest.py
```

検証開始日は2013・2014・2016年の8月1日/12月16日と、2017年4月1日/7月31日の8区間に事前固定しています。検証幅は16暦日です。12月25日は学習CSVに行がないため、年末区間は観測15日を評価します。

2013〜2014年は2013年初、2016〜2017年は2016年初を履歴の起点とし、2015年は学習・検証から除外します。LightGBMは「年代内の履歴を広く使う」「直近365日」の2条件、曜日平均は直近28日で比較します。どの条件も予測起点より後の売上を使いません。年代開始直後56日はラグ作成の準備期間として学習対象から外すため、既存の単一期間評価とは学習条件が一部異なります。

結果は `result/backtest_scores_<日時>.csv`、行別予測は `result/backtest_predictions_<日時>.csv`、比較図と年内パターン図は同じ識別子付きPNGに保存します。年内パターン図は絶対値と年内平均で正規化した値を分け、2017年は8月途中までの集計です。設定・全スコア・コード識別情報は `experiments/backtests/<日時>.json` に保存してGit管理します。単一期間の評価記録と分け、提出モデルの自動選択には混ぜません。

結果の平均・ばらつき・曜日平均への勝敗を確認し、スコアの良い期間だけを選ばないようにしてください。年代ごとの比較は4区間ずつの単純平均です。

年代・季節別の実行結果と検証期間の推奨は[比較結果](docs/backtest-results.md)を参照してください。

## 曜日平均・LightGBM・ランダムフォレストの比較

`storeSales/` 内で実行します。

```sh
uv run --locked python compare_models.py
```

2016-08-01、2016-12-16、2017-04-01、2017-07-31を起点に16暦日を予測します。学習は2016年以降の年代内拡張方式で、最初の56日を準備期間にします。LightGBMとRFは同じ元特徴量と学習行を使い、`log1p(sales)` を学習します。

LightGBMはカテゴリ・欠損を直接処理し、RFは店舗・カテゴリをone-hot化、数値欠損を学習側の中央値で補完します。RFは100本、最大深さ18、最小葉サイズ20、各木の抽出行率30%、特徴量率80%、seed=42の固定設定です。RFだけが使う抽出率・前処理を含めた設定間の比較で、最適化済みモデル同士の比較ではありません。OOBやランダム分割で評価せず、どの方式も同じ未来の全行を採点します。

結果と画像は `result/model_comparison_<日時>.csv/.png`、行別予測は `result/model_comparison_predictions_<日時>.csv` に保存します。条件・平均/全行合算RMSLE・期間ごとのスコア・コードハッシュは `experiments/model_comparisons/<日時>.json` に保存し、Git管理します。これはモデル比較であり、自動の特徴量選択は実施しません。提出モデルの自動選択にも混ぜません。

3方式の固定条件と実行結果は[モデル比較結果](docs/model-comparison.md)を参照してください。
