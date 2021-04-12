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

#Import Library
from sklearn import decomposition
#Assumed you have training and test data set as train and test
# Create PCA obeject 
pca= decomposition.PCA(n_components=6) #default value of k =min(n_sample, n_features)
# For Factor analysis
#fa= decomposition.FactorAnalysis()
# Reduced the dimension of training dataset using PCA
train_reduced = pca.fit_transform(x_train)
#Reduced the dimension of test dataset
test_reduced = pca.transform(x_test)

print('score: %.3f' % pca.score(x_test,y_test))