import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('pima.csv')
df = df.fillna(0) #将缺失值都替换为0
df.head()
# print(len(df))
exc_cols = ['Preg','Plas','Pres','Skin','Insu','Mass','Pedi','Age']
cols = [c for c in df.columns if c not in exc_cols]
print(cols)#['Label']
X = df.loc[:,exc_cols]
y = df['Label'].values
x_train , x_test , y_train ,y_test = train_test_split(X,y)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import joblib
# 模型训练，使用GBDT算法
gbr = GradientBoostingClassifier(n_estimators=3000, max_depth=2, min_samples_split=2, learning_rate=0.1)
gbr.fit(x_train, y_train.ravel())
joblib.dump(gbr, 'train_model_result4.m')   # 保存模型
 
y_gbr = gbr.predict(x_train)
y_gbr1 = gbr.predict(x_test)
acc_train = gbr.score(x_train, y_train)
acc_test = gbr.score(x_test, y_test)
print(acc_train)
print(acc_test)

print('score: %.3f' % gbr.score(x_test,y_test))

