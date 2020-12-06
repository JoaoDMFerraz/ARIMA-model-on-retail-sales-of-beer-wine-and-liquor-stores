# make sure both csv files are in the same folder as this python file
# make sure all these libraries are installed using 'pip install *library*' on the comand line
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from datetime import datetime
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

####################################
############### DATA ###############
####################################

table = pd.DataFrame(pd.read_csv('Retail Sales Beer, Wine, and Liquor Stores.csv'))
table.set_index('DATE', inplace = True)
table.rename(columns={'MRTSSM4453USN':'nominal sales'}, inplace=True)
table['CPI'] = list(pd.DataFrame(pd.read_csv('Consumer Price Index for All Urban Consumers.csv'))['CPIAUCSL_NBD19920101'])
table['CPI'] = table['CPI']/100
table['Population'] = list(pd.DataFrame(pd.read_csv('Population.csv'))['POPTHM'])
table['Population'] = table['Population']/1000
table['real sales'] = table['nominal sales'] / table['CPI']
table['real sales per capita'] = table['real sales'] / table['Population']
table['real sales per capita det12'] = table['real sales per capita'] - table['real sales per capita'].rolling(window = 12).mean()
table['real sales per capita sdiff det12'] = table['real sales per capita det12'] - table['real sales per capita det12'].shift(12)

month_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
month_col = list()
for index in range(table.shape[0]):
    month_col.append(month_list[int(table.index[index][5:7]) - 1])
table.insert(0, 'month', month_col)

#split into train and test data
train_table = table[:320]
test_table = table[320:]

#print the table with all the data
pd.set_option('display.max_columns', None)
print(train_table)
print()

####################################
########## PLOT THE CHARTS #########
####################################

print('############## PLOT THE CHARTS ##########')
#nominal sales & real sales
fig, ax = plt.subplots(figsize=(10, 5))
x_axis = [np.datetime64(train_table.index.values[i]) for i in range(train_table.index.values.shape[0])]
ax.plot(x_axis, train_table['nominal sales'], label = 'Nominal sales')
ax.plot(x_axis, train_table['real sales'], label = 'Real Sales (1992 dollars)')
plt.ylabel('')
plt.title('Retail Sales: Beer, Wine, and Liquor Stores, Not Seasonally Adjusted')
plt.setp(ax.get_xticklabels(), rotation=45)
plt.legend()
plt.show()

#real sales per capita
fig, ax = plt.subplots(figsize=(10, 5))
x_axis = [np.datetime64(train_table.index.values[i]) for i in range(train_table.index.values.shape[0])]
ax.plot(x_axis, train_table['real sales per capita'], label = 'Real Sales per capita (1992 dollars)')
plt.ylabel('')
plt.title('Retail Sales: Beer, Wine, and Liquor Stores, Not Seasonally Adjusted')
plt.setp(ax.get_xticklabels(), rotation=45)
plt.legend()
plt.show()

#real sales per capita with detrending
fig, ax = plt.subplots(figsize=(10, 5))
plt.title('Retail Sales: Beer, Wine, and Liquor Stores, Not Seasonally Adjusted')
x_axis = [np.datetime64(train_table.index.values[i]) for i in range(train_table.index.values.shape[0])]
ax.plot(x_axis, train_table['real sales per capita det12'], label = 'Real Sales (1992 dollars) per capita detrended with SMA12')
plt.setp(ax.get_xticklabels(), rotation=45)
plt.legend()
plt.show()

#real sales per capita with seasonal Diff and detrended
fig, ax = plt.subplots(figsize=(10, 5))
plt.title('Retail Sales: Beer, Wine, and Liquor Stores, Not Seasonally Adjusted')
x_axis = [np.datetime64(train_table.index.values[i]) for i in range(train_table.index.values.shape[0])]
ax.plot(x_axis, train_table['real sales per capita sdiff det12'], label = 'Real Sales (1992 dollars) per capita detrended with SMA12 and seasonal differencing')
plt.setp(ax.get_xticklabels(), rotation=45)
plt.legend()
plt.show()

#seasonality box plot
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x='month',y='real sales per capita det12',data=train_table,ax=ax)
plt.xlabel('')
plt.ylabel('')
plt.title('Retail Sales: Beer, Wine, and Liquor Stores grouped by month')
plt.setp(ax.get_xticklabels(), rotation = 30)
plt.legend()
plt.show()

#Descriptive statistics
print(train_table['real sales per capita sdiff det12'].describe(include = 'all'))
print()

#ADF on real sales per capita
print('ADF on real sales per capita')
result = adfuller(train_table['real sales per capita'].dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#ADF on real sales per capita after detrending
print('ADF on time series after detrending')
result = adfuller(train_table['real sales per capita det12'].dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#Correlograms
fig = tsaplots.plot_pacf(train_table['real sales per capita sdiff det12'].dropna(), lags = 45)
fig = tsaplots.plot_acf(train_table['real sales per capita sdiff det12'].dropna(), lags = 45)
plt.show()

#get AC, PAC, Q-stat and P-value data
acf_df = pd.DataFrame(sm.graphics.tsa.acf(train_table['real sales per capita sdiff det12'].dropna(), qstat=True, alpha=None, fft = True, nlags = 30)).T
acf_df = acf_df.reset_index()
acf_df['index'] = acf_df['index']
acf_df.columns = ['LAG', 'AC', 'Q-stat', 'Prob > Q' ]

pacf_df = pd.DataFrame(sm.graphics.tsa.pacf(train_table['real sales per capita sdiff det12'].dropna(), alpha=None, nlags = 30))
acf_df.insert(2, 'PAC', pacf_df[0])
for col_name in acf_df.columns:
    acf_df[col_name] = round(acf_df[col_name], 4)
print(acf_df)

##################################
######## Model Selection #########
##################################

print('############ Model Selection ##########')
print('Model 1 - ARMA(4,0,3):')
# Model 1 - ARIMA(4,0,3): 
model = ARIMA(endog = train_table['real sales per capita sdiff det12'].dropna(),
              order=(4,0,3),
              missing = 'drop')
model1_fit = model.fit()
print(model1_fit.summary())

print('AR roots')
ar_roots = model1_fit.arroots
print(ar_roots)
print('MA roots')
ma_roots = model1_fit.maroots
print(ma_roots)
123

print('Model 2 - ARMA(4,0,0):')
# Model 2 - ARIMA(4,0,0): 
model = ARIMA(endog = train_table['real sales per capita sdiff det12'].dropna(),
              order=(4,0,0),
              missing = 'drop')
model2_fit = model.fit()
print(model2_fit.summary())

print('AR roots')
ar_roots = model2_fit.arroots
print(ar_roots)
print('MA roots')
ma_roots = model2_fit.maroots
print(ma_roots)

# print('Model 3 - ARMA(0,0,3)'): 
# Model 3 - ARIMA(0,0,3): 
model = ARIMA(endog = train_table['real sales per capita sdiff det12'].dropna(),
              order=(0,0,3),
              missing = 'drop')
model3_fit = model.fit()
print(model3_fit.summary())

print('AR roots')
ar_roots = model3_fit.arroots
print(ar_roots)
print('MA roots')
ma_roots = model3_fit.maroots
print(ma_roots)


# print('Model 4.1 - SARMA(4,0,0)(0,0,2)[12]'):
# Model 4.1 - SARIMA(4,0,0)(0,0,2)[12]:
model = ARIMA(endog = train_table['real sales per capita sdiff det12'].dropna(),
              order = (4,0,0),
              seasonal_order = (0,0,2,12),
              missing = 'drop')
model4_fit = model.fit()
print(model4_fit.summary())

print('AR roots')
ar_roots = model4_fit.arroots
print(ar_roots)
print('MA roots')
ma_roots = model4_fit.maroots
print(ma_roots)


# print('Model 5.1 - SARMA(0,0,3)(0,0,2)[12]:')
# Model 5.1 - SARMA(0,0,3)(0,0,2)[12]:
model = ARIMA(endog = train_table['real sales per capita sdiff det12'].dropna(),
              order = (0,0,3),
              seasonal_order = (0,0,2,12),
              missing = 'drop')
model5_fit = model.fit()
print(model5_fit.summary())

print('AR roots')
ar_roots = model5_fit.arroots
print(ar_roots)
print('MA roots')
ma_roots = model5_fit.maroots
print(ma_roots)

# print('Model 4.2 - SARMA([1,2,3,4,10],0,0)(0,0,2)[12]'):
# Model 4.1 - SARIMA(4,0,0)(0,0,2)[12]:
model = ARIMA(endog = train_table['real sales per capita sdiff det12'].dropna(),
              order = ([1,2,3,4,10],0,0),
              seasonal_order = (0,0,2,12),
              missing = 'drop')
model42_fit = model.fit()
print(model42_fit.summary())

print('AR roots')
ar_roots = model42_fit.arroots
print(ar_roots)
print('MA roots')
ma_roots = model42_fit.maroots
print(ma_roots)

##################################
## Model 4.2. Modifications ######
##################################

print('Model 4.1 - SARMA(4,0,0)(0,0,2)[12]:')
final_model = ARIMA(endog = train_table['real sales per capita sdiff det12'].dropna(),
              order = (4,0,0),
              seasonal_order = (0,0,2,12),
              missing = 'drop')
final_model_fit = final_model.fit()
print(final_model_fit.summary())
final_model_fit.plot_diagnostics(lags = 20)
plt.show()

#get AC, PAC, Q-stat and P-value data
acf_df = pd.DataFrame(sm.graphics.tsa.acf(final_model_fit.resid, qstat=True, alpha=None, fft = True, nlags = 20)).T
acf_df = acf_df.reset_index()
acf_df['index'] = acf_df['index']
acf_df.columns = ['LAG', 'AC', 'Q-stat', 'Prob > Q']

pacf_df = pd.DataFrame(sm.graphics.tsa.pacf(final_model_fit.resid, alpha=None, nlags = 20))
acf_df.insert(2, 'PAC', pacf_df[0])
for col_name in acf_df.columns:
    acf_df[col_name] = round(acf_df[col_name], 4)
print(acf_df)

print('lag 10 on residuals is significant so it is better to only add lag 10 on the model')

print('Model 4.2 - SARMA([1,2,3,4,10],0,0)(0,0,2)[12]:')
final_model = ARIMA(endog = train_table['real sales per capita sdiff det12'].dropna(),
              order = ([1,2,3,4,10],0,0),
              seasonal_order = (0,0,2,12),
              missing = 'drop')
final_model_fit = final_model.fit()
print(final_model_fit.summary())
print('AR roots')
ar_roots = final_model_fit.arroots
print(ar_roots)
print('MA roots')
ma_roots = final_model_fit.maroots
print(ma_roots)

final_model_fit.plot_diagnostics(lags = 20)
plt.show()

#get AC, PAC, Q-stat and P-value data
acf_df = pd.DataFrame(sm.graphics.tsa.acf(final_model_fit.resid, qstat=True, alpha=None, fft = True, nlags = 20)).T
acf_df = acf_df.reset_index()
acf_df['index'] = acf_df['index']
acf_df.columns = ['LAG', 'AC', 'Q-stat', 'Prob > Q']

pacf_df = pd.DataFrame(sm.graphics.tsa.pacf(final_model_fit.resid, alpha=None, nlags = 20))
acf_df.insert(2, 'PAC', pacf_df[0])
for col_name in acf_df.columns:
    acf_df[col_name] = round(acf_df[col_name], 4)
print(acf_df)

####################################
########## Model Forecast ##########
####################################

print('############ model Forecast ##########')
final_model = ARIMA(endog = train_table['real sales per capita sdiff det12'].dropna(),
              order = ([1,2,3,4,10],0,0),
              seasonal_order = (0,0,2,12),
              missing = 'drop')


final_model_fit = final_model.fit()
forecast = final_model_fit.predict(start = 297, end = 321, dynamic = True)


fig, ax = plt.subplots()
x_axis = [np.datetime64(test_table.index.values[i]) for i in range(test_table.index.values.shape[0])]
ax.plot(x_axis, table['real sales per capita sdiff det12'][297:321], label = 'Real Sales (1992 dollars) per capita with seasonal differencing (m = 12)')
ax.plot(x_axis, forecast[1:], label = 'forecast')
plt.setp(ax.get_xticklabels(), rotation=45)
plt.legend()
plt.show()
