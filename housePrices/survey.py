import matplotlib.pyplot as plt
import seaborn as sns

# 学習データの読み込み
from data_import import train, test

# 売却価格のヒストグラム
sns.histplot(train['SalePrice'])
plt.savefig('SalePrice.png')

# 売却価格の概要
print(train["SalePrice"].describe())
print(f"歪度: {round(train['SalePrice'].skew(),4)}" )
print(f"尖度: {round(train['SalePrice'].kurt(),4)}" )


# 学習データの欠損状況
deficiency_train = train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)
# テストデータの欠損状況
deficiency_test = test.isnull().sum()[test.isnull().sum()>0].sort_values(ascending=False)

# 欠損を含むカラムをリスト化
na_col_list_train = train.isnull().sum()[train.isnull().sum()>0].index.tolist()
na_col_list_test = test.isnull().sum()[test.isnull().sum()>0].index.tolist()

# データ型を確認
no_col_dtype_train = train[na_col_list_train].dtypes.sort_values()
no_col_dtype_test = test[na_col_list_test].dtypes.sort_values()
print('学習データの欠損状況\n',deficiency_train)
print('テストデータの欠損状況\n',deficiency_test)
print('欠損学習データのデータ型\n',no_col_dtype_train)
print('欠損訓練データのデータ型\n',no_col_dtype_test)
