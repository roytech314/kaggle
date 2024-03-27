# model_training_evaluation.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 訓練データとテストデータをインポート
from data_import import train, test

# データ前処理と特徴量エンジニアリング

# 名前の分割
def split_name(name_str):
    last_name, title_and_name = name_str.split(", ", 1)
    title, first_name = title_and_name.split(" ", 1)
    title = title.replace(".", "")
    return pd.Series([last_name, title, first_name])

train[["Name_1", "Name_2", "Name_3"]] = train["Name"].apply(split_name)
test[["Name_1", "Name_2", "Name_3"]] = test["Name"].apply(split_name)

# 欠損値の補完
train["Age"] = train["Age"].fillna(train["Age"].mean())
train["Embarked"] = train["Embarked"].fillna("S")
train["Fare"] = train["Fare"].fillna(train["Fare"].mean())
test["Age"] = test["Age"].fillna(train["Age"].median())
test["Embarked"] = test["Embarked"].fillna("S")
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

# 文字列データを数値に変換
train.replace({"Sex": {"male": 0, "female": 1}, "Embarked": {"S": 0, "C": 1, "Q": 2}}, inplace=True)
test.replace({"Sex": {"male": 0, "female": 1}, "Embarked": {"S": 0, "C": 1, "Q": 2}}, inplace=True)

# SibSpとParchを結合してFamilySizeを計算
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

# Alone_Flagを計算
train["Alone_Flag"] = (train["FamilySize"] == 1).astype(int)
test["Alone_Flag"] = (test["FamilySize"] == 1).astype(int)

# 特徴量の選択
feature_columns = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Alone_Flag"]
train_x = train[feature_columns].values
train_y = train["Survived"].values
test_x = test[feature_columns].values

# 訓練データとテストデータを分割
X_train, X_test, Y_train, Y_test = train_test_split(
    train_x, train_y, test_size=0.3, random_state=0
)

# モデルのパラメータチューニング
tuned_parameters = {
    "n_estimators": np.arange(95, 106, 1),
    "max_depth": np.arange(6, 9, 1),
    "min_samples_split": np.arange(3, 6, 1),
}

gridSearch = GridSearchCV(
    estimator=RandomForestClassifier(criterion="gini", random_state=0, verbose=1),
    param_grid=tuned_parameters,
    cv=5,
)

gridSearchFit = gridSearch.fit(X_train, Y_train)

# 最適なパラメータを用いたランダムフォレストモデルの構築
model = RandomForestClassifier(
    bootstrap=True,
    n_estimators=gridSearchFit.best_params_["n_estimators"],
    max_features="sqrt",
    criterion="gini",
    max_depth=gridSearchFit.best_params_["max_depth"],
    min_samples_split=gridSearchFit.best_params_["min_samples_split"],
    min_samples_leaf=1,
    random_state=0,
)

model.fit(X_train, Y_train)

# モデルの評価
print("Train score = ", model.score(X_train, Y_train))
# モデルの評価
print("Test score = ", model.score(X_test, Y_test))

# 訓練データに対する予測とその評価
pred_train = model.predict(X_train)
print("\nClassification report for training data:\n", classification_report(Y_train, pred_train))

# テストデータに対する予測とその評価
pred_test = model.predict(X_test)
print("\nClassification report for test data:\n", classification_report(Y_test, pred_test))

# 特徴量の重要度
print("\nFeature importances:")
for name, score in zip(feature_columns, model.feature_importances_):
    print(f"{name}: {score:.2f}")
