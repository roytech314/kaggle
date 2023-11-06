import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")


def split_name(name_str):
    # カンマで分割し、姓と名前の部分を取得
    last_name, title_and_name = name_str.split(", ", 1)

    # 最初のスペースで分割し、称号と名前の部分を取得
    title, first_name = title_and_name.split(" ", 1)

    # 称号からピリオドを除去
    title = title.replace(".", "")

    return pd.Series([last_name, title, first_name])


# Nameカラムの各要素に対してsplit_name関数を適用し、結果を新しいカラムに追加
train[["Name_1", "Name_2", "Name_3"]] = train["Name"].apply(split_name)
test[["Name_1", "Name_2", "Name_3"]] = test["Name"].apply(split_name)

# 欠損値補完
train["Age"] = train["Age"].fillna(train["Age"].mean())

# 年齢範囲の境界を定義
bins = [0, 19, 59, 100]

# カテゴリのラベルを定義
labels = [0, 1, 2]

# cut関数を使用して年齢をカテゴリに分ける
train["Age_Category"] = pd.cut(train["Age"], bins=bins, labels=labels, right=False)

train["Embarked"] = train["Embarked"].fillna("S")
train["Fare"] = train["Fare"].fillna(train["Fare"].mean())
# train["Cabin"] = train["Cabin"].fillna("Z")

test["Age"] = test["Age"].fillna(train["Age"].median())

# cut関数を使用して年齢をカテゴリに分ける
test["Age_Category"] = pd.cut(test["Age"], bins=bins, labels=labels, right=False)


test["Embarked"] = test["Embarked"].fillna("S")
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
# test["Cabin"] = test["Cabin"].fillna("Z")

# 訓練データ文字列⇒数値変換
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2


# テストデータ文字列⇒数値変換
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1
test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

# SibSpとParchを結合
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1


for train in [train]:
    train["Alone_Flag"] = 0
    train.loc[train["FamilySize"] == 1, "Alone_Flag"] = 1

for test in [test]:
    test["Alone_Flag"] = 0
    test.loc[test["FamilySize"] == 1, "Alone_Flag"] = 1

# Cabinの最初の１文字を取得
train["Cabin_alpha"] = train["Cabin"].str[:1]
test["Cabin_alpha"] = test["Cabin"].str[:1]

# アルファベットを数値に変換する辞書を定義
cabin_mapping = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "T": 8,  # もし'T'が存在する場合
    "Z": 0,  # 欠損値を'Z'で埋めているので、'Z'もマッピング
}

# Cabin_alphaカラムの値を数値に変換
# train["Cabin_alpha"] = train["Cabin_alpha"].replace(cabin_mapping)
# test["Cabin_alpha"] = test["Cabin_alpha"].replace(cabin_mapping)

train["Cabin_Flag"] = train["Cabin"].notnull().astype(int)
test["Cabin_Flag"] = test["Cabin"].notnull().astype(int)


def ticket_alpha(df):  # Ticketアルファベット
    s = df["Ticket"]
    if s.isnumeric():  # 数字
        result = "ZZZ"  # 数字のみのチケット
    else:
        rtn = re.findall(r"\w+", s)  # アルファベット、アンダーバー、数字（複数の時は配列）
        rslt = ""
        if len(rtn) > 1:
            for i in range(len(rtn) - 1):  # 最後の配列は数字なので入れない
                rslt += rtn[i]
        else:
            rslt = rtn[0]  # 英字だけのTicket
        result = rslt
    return result


def ticket_num(df):  # Ticket数字
    s = df["Ticket"]
    if s.isnumeric():  # 数字
        result = int(s)
    else:
        rtn = re.findall(r"\d+", s)  # 数字部分を取り出し（複数の時は配列）
        if len(rtn) > 0:
            result = int(rtn[len(rtn) - 1])  # 複数取り出した時は最後を使う
        else:
            result = int(999999)  # 英字のみのTicketは999999をセット
    return result


train["Ticket_Alpha"] = train.apply(ticket_alpha, axis=1)
train["Ticket_Num"] = train.apply(ticket_num, axis=1)
test["Ticket_Alpha"] = test.apply(ticket_alpha, axis=1)
test["Ticket_Num"] = test.apply(ticket_num, axis=1)

le = LabelEncoder()  # LabelEncoderのインスタンスを作成
train["Ticket_Alpha_encoded"] = le.fit_transform(
    train["Ticket_Alpha"]
)  # Ticket_alphaのエンコード


def safe_label_encoder_transform(label_encoder, labels):
    # classes_ 属性から既知のラベルのセットを取得
    known_labels = set(label_encoder.classes_)

    # 未知のラベルは-1、既知のラベルはそのままエンコード
    return [
        label_encoder.transform([label])[0] if label in known_labels else -1
        for label in labels
    ]


test["Ticket_Alpha_encoded"] = safe_label_encoder_transform(le, test["Ticket_Alpha"])


train["Ticket_Converted"] = (
    train["Ticket_Alpha_encoded"] * 10**7 + train["Ticket_Num"]
)
test["Ticket_Converted"] = test["Ticket_Alpha_encoded"] * 10**7 + test["Ticket_Num"]

name_mapping = {
    # 高級
    "Don": 0,
    "Dona": 0,
    "Lady": 0,
    "Sir": 0,
    "Jonkheer": 0,
    "the": 0,
    # 軍人
    "Major": 0,
    "Col": 0,
    "Capt": 0,
    # 一般
    "Mr": 1,
    "Mrs": 2,
    "Miss": 3,
    "Master": 4,
    "Mme": 5,
    "Ms": 6,
    "Mlle": 7,
    # 専門家
    "Rev": 8,
    "Dr": 8,
}

train["Name_2"] = train["Name_2"].map(name_mapping)
test["Name_2"] = test["Name_2"].map(name_mapping)


# trainの目的変数と説明変数の値を取得
train_x = train[
    [
        "Pclass",
        "Sex",
        "Age",
        # "Age_Category",
        "Fare",
        "Name_2",
        # "Cabin_alpha",
        "Alone_Flag",
        # "Cabin_Flag",
        # "Embarked",
        "FamilySize",
        "Ticket_Converted",
    ]
].values
train_y = train["Survived"].values

# testの説明変数の値を取得
test_x = test[
    [
        "Pclass",
        "Sex",
        "Age",
        # "Age_Category",
        "Fare",
        "Name_2",
        # "Cabin_alpha",
        "Alone_Flag",
        # "Cabin_Flag",
        # "Embarked",
        "FamilySize",
        "Ticket_Converted",
    ]
].values

# 訓練データとテストデータを分割
X_train, X_test, Y_train, Y_test = train_test_split(
    train_x, train_y, test_size=0.3, random_state=0
)

# 学習に関するパラメータ設定
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

print("Grid Search Best parameters:", gridSearch.best_params_)
print("Grid Search Best validation score:", gridSearch.best_score_)
print(
    "Grid Search Best training score:",
    gridSearch.best_estimator_.score(train_x, train_y),
)

pred_x = gridSearch.best_estimator_.predict(train_x)
print("\n", classification_report(train_y, pred_x))


# ランダムフォレストモデルの作成
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


# 特徴量の名前リスト
feature_names = [
    "Pclass",
    "Sex",
    "Age",
    # "Age_Category",
    "Fare",
    "Name_2",
    # "Cabin_alpha",
    "Alone_Flag",
    # "Cabin_Flag",
    # "Embarked",
    "FamilySize",
    "Ticket_Converted",
]

model.fit(X_train, Y_train)
# 特徴量の重要度を表示
for name, score in zip(feature_names, model.feature_importances_):
    print(name, "  {:.2f} %".format(score * 100))

# 訓練データスコア
print("train score = ", model.score(X_train, Y_train))

# テストデータスコア
print("test score = ", model.score(X_test, Y_test))

# 全データ評価
print("score = ", model.score(train_x, train_y))

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
