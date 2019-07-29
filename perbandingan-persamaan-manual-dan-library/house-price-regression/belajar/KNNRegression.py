# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load data
df_train = pd.read_csv('train.csv')
# print(df_train)

# take sample feature
sample_df_train = df_train[[
  'LotArea',
  'GarageArea',
  'SalePrice'
]]
# print(sample_df_train.head())

# clean data
sample_df_train = sample_df_train.dropna()

y_sample = sample_df_train['SalePrice']
sample_df_train = sample_df_train.drop(
  'SalePrice', 
  axis = 1
)

'''
NORMALIZATION
Formula Transformation
X_std = (X - X.min(axis=0)) / (X.max(axis=0)) - X.min(axis=0)
X_scaled = X_std * (max - min) + min
nb: max and min given by feature_range
'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1),copy=True)
x_scaled = scaler.fit_transform(sample_df_train)

x_sample = pd.DataFrame(x_scaled)
sampled = pd.merge(x_sample,y_sample,right_index = True,left_index = True)

'''
Create Train and Test Set from train.csv
'''
from sklearn.model_selection import train_test_split
train, test = train_test_split(sampled, test_size = 0.3)
# print(train)
# print(test)
x_train = train.drop('SalePrice', axis = 1)
y_train = train['SalePrice']

x_test = test.drop('SalePrice', axis = 1)
y_test = test['SalePrice']

'''
Predict
import suport library
'''
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt

K = 3

# make the model
model = neighbors.KNeighborsRegressor(n_neighbors = K)

# fitting
model.fit(x_train,y_train)

# prediction
prediction = model.predict(x_test)
# print(prediction)

'''
Check Error
'''
error = sqrt(mean_squared_error(y_test,prediction))
# print(error)

'''
Check error with some
'''
all_error = []
str_error = []
for K in range(100):
  K += 1
  model = neighbors.KNeighborsRegressor(n_neighbors = K)

  # fitting
  model.fit(x_train,y_train)

  # prediction
  prediction = model.predict(x_test)

  # count error and store it
  error = sqrt(mean_squared_error(y_test,prediction))

  all_error.append(error)
  str_ = "RMSE for K = " + str(K) + " is " + str(error)
  str_error.append(str_)
# print(str_error)

'''
Plotting
'''
plot_curve = pd.DataFrame(all_error)
plot_curve.plot()

mini = all_error[0]
index = 0
for i in range(len(all_error)):
  if all_error[i] < mini:
    mini = all_error[i]
    index = i + 1
# print("The smallest value RMSE is ", mini, " in index ", index)

'''
predic test.csv
load data test
'''
df_test = pd.read_csv('test.csv')
df_test = df_test[['LotArea','GarageArea']]
df_test = df_test.dropna()
# print(df_test)
# predict

'''
Normalization
'''
test_scaled = scaler.fit_transform(df_test)
# print(test_scaled)

test_scaled = pd.DataFrame(test_scaled)
# print(test_scaled)

K = index
model = neighbors.KNeighborsRegressor(n_neighbors=K)
model.fit(x_train,y_train)

predict = model.predict(test_scaled)
print(predict)

pd_predict = pd.DataFrame(predict)
result = pd.merge(df_test,pd_predict,right_index = True,left_index = True)
result.rename(columns = {0:"SalePrice"}, inplace = True)
print(result)