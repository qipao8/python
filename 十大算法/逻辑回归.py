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
X_train , X_test , y_train ,y_test = train_test_split(X,y)

#Import Library
from sklearn.linear_model import LogisticRegression
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create logistic regression object
model = LogisticRegression()
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Equation coefficient and Intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)
#Predict Output
predicted= model.predict(X_test)

print('score: %.3f' % model.score(X_test,y_test))