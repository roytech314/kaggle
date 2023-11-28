# prediction_and_saving.py

import datetime
import pandas as pd
import numpy as np

# data_initialization.pyからtestデータをインポート
from data_initialization import test

# model_training_evaluation.pyからモデルをインポート
from model_training_evaluation import model

# テストデータの特徴量を選択
feature_columns = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Alone_Flag"]
test_x = test[feature_columns].values

# テストデータの予測
test_predict = model.predict(test_x)

# 結果をCSV出力
t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, "JST")
now = datetime.datetime.now(JST)
d = now.strftime("%Y%m%d%H%M%S")

PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(test_predict, PassengerId, columns=["Survived"])
my_solution.to_csv(f"./results/randam_forest_{d}.csv", index_label=["PassengerId"])
