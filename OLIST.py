#!/usr/bin/env python
# coding: utf-8

# # SALES PREDICTION USING ARIMA AND PROPHET FOR BRAZILIAN E-COMMERCE

# The dataset was generously provided by Olist, the largest department store in Brazilian marketplaces. Olist connects small businesses from all over Brazil to channels without hassle and with a single contract. Those merchants are able to sell their products through the Olist Store and ship them directly to the customers using Olist logistics partners. See more on our website: www.olist.com
# 
# After a customer purchases the product from Olist Store a seller gets notified to fulfill that order. Once the customer receives the product, or the estimated delivery date is due, the customer gets a satisfaction survey by email where he can give a note for the purchase experience and write down some comments.

# **What is ARIMA model?**
# 
# A. ARIMA(Auto Regressive Integrated Moving Average) is a combination of 2 models AR(Auto Regressive) & MA(Moving Average). It has 3 hyperparameters - P(auto regressive lags),d(order of differentiation),Q(moving avg.) which respectively comes from the AR, I & MA components. The AR part is correlation between prev & current time periods. To smooth out the noise, the MA part is used. The I part binds together the AR & MA parts.

# # 📤 IMPORT LIBRARIES

# In[ ]:


get_ipython().system('pip install statsmodels')
get_ipython().system('pip install pmdarima')
get_ipython().system('conda install -c conda-forge xgboost')


# In[ ]:


import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
from scipy import stats
import os
import itertools

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from pylab import rcParams

from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

from xgboost import  XGBRegressor

# elimina os warnings das bibliotecas
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # 💾 CHECK OUT THE DATA

# In[ ]:


df_item = pd.read_csv("Documents/Mentoria_stack/dataset/olist_order_items_dataset.csv")
df_reviews = pd.read_csv("Documents/Mentoria_stack/dataset/olist_order_reviews_dataset.csv")
df_orders = pd.read_csv("Documents/Mentoria_stack/dataset/olist_orders_dataset.csv")
df_products = pd.read_csv("Documents/Mentoria_stack/dataset/olist_products_dataset.csv")
df_geolocation = pd.read_csv("Documents/Mentoria_stack/dataset/olist_geolocation_dataset.csv")
df_sellers = pd.read_csv("Documents/Mentoria_stack/dataset/olist_sellers_dataset.csv")
df_order_pay = pd.read_csv("Documents/Mentoria_stack/dataset/olist_order_payments_dataset.csv")
df_customers = pd.read_csv("Documents/Mentoria_stack/dataset/olist_customers_dataset.csv")
df_category = pd.read_csv("Documents/Mentoria_stack/dataset/product_category_name_translation.csv")


# **ALL IN ONE**

# In[ ]:


# merge all the tables in one dataframe
df_train = df_orders.merge(df_item, on='order_id', how='left')
df_train = df_train.merge(df_order_pay, on='order_id', how='outer', validate='m:m')
df_train = df_train.merge(df_reviews, on='order_id', how='outer')
df_train = df_train.merge(df_products, on='product_id', how='outer')
df_train = df_train.merge(df_customers, on='customer_id', how='outer')
df_train = df_train.merge(df_sellers, on='seller_id', how='outer')


# **CONVERT DATE COLUMNS TO TIMESTAMP**

# In[ ]:


df_train['order_purchase_time'] = pd.to_datetime(df_train['order_purchase_timestamp']).dt.time


# In[ ]:


# Extracting attributes for purchase date - Date
df_train['order_purchase_timestamp'] = pd.to_datetime(df_train['order_purchase_timestamp']).dt.date
df_train['order_delivered_customer_date'] = pd.to_datetime(df_train['order_delivered_customer_date']).dt.date
df_train['order_estimated_delivery_date'] = pd.to_datetime(df_train['order_estimated_delivery_date']).dt.date
df_train['order_approved_at'] = pd.to_datetime(df_train['order_approved_at']).dt.date
df_train['order_delivered_carrier_date'] = pd.to_datetime(df_train['order_delivered_carrier_date']).dt.date


# In[ ]:


# Extracting attributes for purchase date - Year and Month
df_train['order_purchase_year'] = pd.to_datetime(df_train['order_purchase_timestamp']).dt.year
df_train['order_purchase_month'] = pd.to_datetime(df_train['order_purchase_timestamp']).dt.month
df_train['order_purchase_month_name'] = df_train['order_purchase_timestamp'].apply(lambda x: x.strftime('%b'))


# In[ ]:


# Extracting attributes for purchase date - Day and Day of Week
df_train['order_purchase_day'] = pd.to_datetime(df_train['order_purchase_timestamp']).apply(lambda x: x.day)
df_train['order_purchase_dayofweek'] = pd.to_datetime(df_train['order_purchase_timestamp']).apply(lambda x: x.dayofweek)
df_train['order_purchase_dayofweek_name'] = pd.to_datetime(df_train['order_purchase_timestamp']).dt.day_name()


# In[ ]:


df_train['order_purchase_timestamp'] = pd.to_datetime(df_train['order_purchase_timestamp'])
df_train['order_purchase_time'] = pd.to_datetime(df_train['order_purchase_time'],format='%H:%M:%S').dt.time
df_train['order_delivered_customer_date'] = pd.to_datetime(df_train['order_delivered_customer_date'])
df_train['order_estimated_delivery_date'] = pd.to_datetime(df_train['order_estimated_delivery_date'])


# In[ ]:


df_train.head()


# In[ ]:


df_train_state=df_train


# In[ ]:


df_train.info()


# CLEAN DATA

# In[ ]:


df_train.shape


# <iframe src="https://www.kaggle.com/embed/fekmea/preparation-olist-dataset/notebook?cellIds=10&kernelSessionId=80861728" height="300" style="margin: 0 auto; width: 100%; max-width: 950px;" frameborder="0" scrolling="auto" title="Preparation Olist dataset"></iframe>

# In[ ]:


df_train.isnull().sum().sort_values(ascending = False).head()


# In[ ]:


# We may drop the review_comment_title column, as all values are null
df_train.drop(['review_comment_title'], axis=1, inplace=True)


# Default review comment message

# In[ ]:


df_train['review_comment_message'] = df_train['review_comment_message'].fillna('No message')


# In[ ]:


df_train.isnull().sum().sort_values(ascending = False)


# Drop nan values

# In[ ]:


df_train_0 = df_train.shape[0]
#Remove missing values with dropna
df_train= df_train.dropna()
df_train_1 = df_train.shape[0]
print(f'{round(((df_train_0-df_train_1)/df_train_1)*100,2)}% nan values points were eliminated')


# # VISUALIZATION

# # Price Distribution

# In[ ]:


# Getting the histogram and normal probability plot
plt.figure(figsize=(16,12))
plt.suptitle('Price Distributions', fontsize=22)
plt.subplot(221)
g = sns.histplot(df_train['price'], kde=True)
g.set_title("Price Distributions", fontsize=18)
g.set_xlabel("Price Values", fontsize=15)
g.set_ylabel("Probability", fontsize=15)

plt.subplot(222)
g1 = sns.histplot(np.log(df_train['price']), kde=True)
g1.set_title("Price(LOG) Distributions", fontsize=18)
g1.set_xlabel("Price Values", fontsize=15)
g1.set_ylabel("Probability", fontsize=15)

plt.show()

plt.figure(figsize=(16,12))

plt.suptitle('Price Distribution Probability Plot', fontsize=22)

plt.subplot(221)
res = stats.probplot(df_train['price'], plot=plt, fit=True, rvalue=True);


plt.subplot(222)
res = stats.probplot(np.log(df_train['price']), plot=plt, fit=True, rvalue=True);

plt.show()


# **Price Distribution:**
# Histogram of a sample from a right-skewed distribution – it looks unimodal and skewed right.
# 
# **Price Log Distribution:**
# Histogram of a sample from a normal distribution – it looks fairly symmetric and unimodal.
# 
# **Probability Plot - Price Distribution:**
# Normal probability plot of a sample from a right-skewed distribution – it has an inverted C shape.
# 
# **Probability Plot - Price Log Distribution:**
# Normal probability plot of a sample from a normal distribution – it looks fairly straight, at least when the few large and small values are ignored.

# Let's Check the Features Through the Time

# Our current timestamps can be tricky to work with, so we'll be using the average daily price amount for that month, and we're using the start of each month as the timestamp.

# In[ ]:


df_train['order_purchase_timestamp'].min(), df_train['order_purchase_timestamp'].max()


# In[ ]:


df_train = df_train.groupby('order_purchase_timestamp')['price'].sum().reset_index()
df_train


# **Indexing with Time Series of Data**

# In[ ]:


df_train=df_train.set_index('order_purchase_timestamp')
df_train.index


# In[ ]:


sales = df_train['price'].resample('MS').mean()
sales2=sales[3:]
sales


# # **Viewing Furniture Sales Time Series Data**

# In[ ]:


sales.plot(figsize=(15, 6))
plt.xlabel('Date')
plt.ylabel('Sell per month')
plt.show()


# In[ ]:


#Determine rolling statistics
sales_mean = sales.rolling(window=12).mean() #window size 12 denotes 12 months, giving rolling mean at yearly level
sales_std = sales.rolling(window=12).std()
print(sales_mean,sales_std)


# In[ ]:


#Plot rolling statistics
plt.figure(figsize=(20,10))
orig = plt.plot(sales, color='blue', label='Original')
sales_mean = plt.plot(sales_mean, color='red', label='Rolling Mean')
sales_std = plt.plot(sales_std, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)


# In[ ]:


# Primeiro, vamos decompor a série pra avaliar tendência
# Sazonalidade e resíduo
from statsmodels.tsa.seasonal import seasonal_decompose

resultado = seasonal_decompose(df_train)

fig = plt.figure(figsize=(8, 6))  
fig = resultado.plot()


# Teste de estacionariedade

# In[ ]:


# Teste de estacionariedade. 
# A hipótese nula é que a série não é estacionária
# Ou seja, se o p-valor for menor que 0,05, rejeitamos
# que a série não é estacionária. Caso seja maior, não podemos
# descartar que a série não é estacionária
from statsmodels.tsa.stattools import adfuller

result=adfuller(df['producao'].dropna())
print(f'Teste ADF:{result[0]}')
print(f'p-valor:{result[1]}')


# # ARIMA Model for Time Series Forecasting

# Let's apply one of the most used methods for forecasting time series, known as ARIMA, which stands Autoregressive Integrated Moving Average.
# 
# ARIMA models are denoted with the ARIMA notation (p, d, q). These three parameters are responsible for the seasonality, trend and noise in the data:

# In[ ]:


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[ ]:


warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(sales,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            #enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue           


# In[ ]:


mod = sm.tsa.statespace.SARIMAX(sales2,
                                order=(1, 1, 0),
                                seasonal_order=(0, 1, 0, 12),
                                enforce_invertibility=False)
results = mod.fit(disp=False)
print(results.summary().tables[1])


# In[ ]:


results.plot_diagnostics(variable=0, lags=2, figsize=10)
plt.show()


# A saída acima sugere que o SARIMAX (0, 1, 0) x (1, 1, 0, 12) produz o menor valor de AIC de 4.0. Portanto, devemos considerar isso como a melhor opção.

# # Adjusting the ARIMA model

# In[ ]:


from statsmodels.tsa.arima.model import ARIMA
from pandas import DataFrame
# fit model
model = ARIMA(sales, order=(1,1,1))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
# density plot of residuals
residuals.plot(kind='kde')
plt.show()
# summary stats of residuals
print(residuals.describe())


# **Customer's State Distribution**

# In[ ]:


df_train_state


# In[ ]:


plt.figure(figsize=(35,14))

plt.subplot(221)
sns.set(font_scale=1.5) 
g2 = sns.boxplot(x='customer_state', y='price', 
                 data=df_train_state[df_train_state['price'] != -1])
g2.set_title("Customer's State by Price", fontsize=20)
g2.set_xlabel("State", fontsize=20)
g2.set_ylabel("Price", fontsize=20)
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)

plt.show()


# In[ ]:


plt.figure(figsize=(30,14))
df_train_state['price_log'] = np.log(df_train_state['price'])

plt.subplot(221)
sns.set(font_scale=1.5) 
g2 = sns.boxplot(x='customer_state', y='price_log', 
                 data=df_train_state[df_train_state['price'] != -1])
g2.set_title("Customer's State by Price Log", fontsize=20)
g2.set_xlabel("State", fontsize=20)
g2.set_ylabel("Price Log", fontsize=20)
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)

plt.show()


# In[ ]:


plt.figure(figsize=(16,12))

g = sns.countplot(x='customer_state', data=df_train_state, orient='h', order=df_train['customer_state'].value_counts().index)
g.set_title("DISTRIBUIÇÃO DE CLIENTES POR ESTADO", fontsize=20)
g.set_xlabel("Estado", fontsize=17)
g.set_ylabel("Quantidade (%)", fontsize=17)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
sizes = []
total=len(df_train_state)

for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=14) 
g.set_ylim(0, max(sizes) * 1.1)

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# https://github.com/letsdata/series-temporais-python/blob/main/series-temporais-python.ipynb
