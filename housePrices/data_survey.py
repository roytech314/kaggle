import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 学習データの読み込み
from data_import import train

# 売却価格のヒストグラム
# 以下2行をコメントアウトして実行するとpngファイルが生成される
sns.histplot(train['SalePrice'])
plt.savefig('SalePrice.png')

# 売却価格の概要
print(train["SalePrice"].describe())
print(f"歪度: {round(train['SalePrice'].skew(),4)}" )
print(f"尖度: {round(train['SalePrice'].kurt(),4)}" )
