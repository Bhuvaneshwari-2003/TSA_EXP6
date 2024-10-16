# Ex.No: 6               HOLT WINTERS METHOD
### Date: 



### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

df=pd.read_csv('dailysales.csv',parse_dates=['date'])
df.info()
df.head()
df.isnull().sum()

df=df.groupby('date').sum()
df.head(10)
df=df.resample(rule='MS').sum()
df.head(10)
df.plot()

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

seasonal_decompose(df,model='additive').plot();
train=df[:19] #till Jul19
test=df[19:] # from aug19
train.tail()
test

from statsmodels.tsa.holtwinters import ExponentialSmoothing
hwmodel=ExponentialSmoothing(train.sales,trend='add', seasonal='mul', seasonal_periods=4).fit()

test_pred=hwmodel.forecast(5)
test_pred
train['sales'].plot(legend=True, label='Train', figsize=(10,6))
test['sales'].plot(legend=True, label='Test')
test_pred.plot(legend=True, label='predicted_test')

from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(test,test_pred))
df.sales.mean(), np.sqrt(df.sales.var())

final_model=ExponentialSmoothing(df.sales,trend='add', seasonal='mul', seasonal_periods=4).fit()

pred=final_model.forecast(10)
pred
df['sales'].plot(legend=True, label='sales', figsize=(10,6))
pred.plot(legend=True, label='prediction')
```

### OUTPUT:
SALES PLOT :

![image](https://github.com/user-attachments/assets/c409af15-1f34-464f-831a-1d054d291b4a)

SEASONAL DECOMPOSING (ADDITIVE) :

![image](https://github.com/user-attachments/assets/15e893fe-1afa-412f-92be-ced7f8c570ac)


TEST_PREDICTION

![image](https://github.com/user-attachments/assets/05a12483-29a5-4f4d-b536-2b3da5e2aced)


FINAL_PREDICTION

![image](https://github.com/user-attachments/assets/a1f3b9d9-1eba-4679-bde0-a17aa21bdf24)


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
