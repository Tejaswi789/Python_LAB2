import pandas as pd
import copy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
advertising=pd.read_csv('winequality-white.csv')
print(advertising.corr())
train,test=train_test_split(advertising,test_size=0.2)#splitting data into training and test data
train_label=train['quality']
test_label=test['quality']
#train_eda=copy.deepcopy(train)
train_eda=train.dropna(how='any',axis=0)
train_eda_label=train_eda['quality']#removing null data
train=train.drop(columns=['quality'])
train_eda=train_eda.drop(columns=['quality'])
test=test.drop(columns=['quality'])
clf1=LinearRegression()
clf1.fit(train,train_label) #fitting Linear regression without EDA
clf2=LinearRegression()
clf2.fit(train_eda,train_eda_label)
answer=clf1.predict(test)
answer1=clf2.predict(test)
mean_squared_error_eda=mean_squared_error(test_label,answer1)
r2_score_eda=r2_score(test_label,answer1)
mean_squared_error1 = mean_squared_error(test_label, answer)
r2_score1 = r2_score(test_label,answer)
print("mean squared error before applying EDA is :",mean_squared_error1)
print("R2 score before applying EDA is :",r2_score1)
print("mean Squared error after applyinf EDA is : ",mean_squared_error_eda)
print("R2 score after applying EDA is : ",r2_score_eda)