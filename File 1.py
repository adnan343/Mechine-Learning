#Load Data:

import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/california_housing_train.csv")
print(df)


#Data Seperation as x and y:

y = df['median_house_value']
print(y)

x = df.drop('median_house_value', axis=1)
print(x)


#Data Splitting:

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=100)
print(x_train)
print(x_test)


#Model Building:

#Linear Regression:

#Training the model:
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

#Applying the model to make a prediction:
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)
print(y_lr_train_pred)
print(y_lr_test_pred)


#Evaluate model performance:

# print(y_train)
# print(y_lr_train_pred)

from sklearn.metrics import mean_squared_error, r2_score
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

print('LR MSE (train): ', lr_train_mse)
print('LR R2 (train): ', lr_train_r2)
print('LR MSE (test): ', lr_test_mse)
print('LR R2 (test): ', lr_test_r2)

lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print(lr_results)


#Random Forest:

#Training the model:
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)


#Applying the model to make a prediction:
y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)
print(y_rf_train_pred)
print(y_rf_test_pred)


#Evaluate model performance:

# print(y_train)
# print(y_lr_train_pred)

from sklearn.metrics import mean_squared_error, r2_score
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

print('RF MSE (train): ', rf_train_mse)
print('RF R2 (train): ', rf_train_r2)
print('RF MSE (test): ', rf_test_mse)
print('RF R2 (test): ', rf_test_r2)

rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print(rf_results)


# Model comparison:

df_models = pd.concat([lr_results, rf_results], axis=0)
print()

df_models.reset_index(drop=True)
print(df_models)


#Data Visualization of prediction results:

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5,5))
plt.scatter(x=y_train, y=y_lr_train_pred, alpha=0.3)
z=np.polyfit(y_train, y_lr_train_pred, 1)
p=np.poly1d(z)
plt.plot(y_train, p(y_train), '#F8766D')
plt.ylabel('Predicted Median House Value using linear regression')
plt.xlabel('Experimental Median House Value')
plt.show()


plt.figure(figsize=(5,5))
plt.scatter(x=y_train, y=y_rf_train_pred, alpha=0.3)
z=np.polyfit(y_train, y_rf_train_pred, 1)
p=np.poly1d(z)
plt.plot(y_train, p(y_train), '#F8766D')
plt.ylabel('Predicted Median House Value using random forest')
plt.xlabel('Experimental Median House Value')
plt.show()

