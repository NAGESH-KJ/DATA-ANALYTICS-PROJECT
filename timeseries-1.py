
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("D:\DA_ASSIGNMENT\PROJECT\project.csv",parse_dates =["date"],index_col ="date")


# In[2]:


#dropping the columns which are unnecessary.
df1=df.drop(columns=['id','month','hour'])


# In[4]:


#monthly crime trend near lamp-posts across years
import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
#resampling the dataset monthly across years and replacing it with mean.
df_monthly = df1.lampdist.resample('M').mean()
#decompse the time series into trend,seasonality and residual
result = seasonal_decompose(df_monthly)
#df_monthly
result.plot()
pyplot.show()


# In[6]:


#without resampling for the original data adf test is conducted to chech if the time series is stationary or not
from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
X = df1['lampdist'].values
#print(X)
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# In[7]:


diff = list()
for i in range(1, len(df_monthly)):
    value = df_monthly[i] - df_monthly[i - 1]
    diff.append(value)
pyplot.plot(diff)
pyplot.show()


# In[ ]:


#Since the adf test ensured that the time series is either stationary or trend stationary remove the trend gives us the above
#plot which tells that the series is trend stationary since removing the trend made the series a stationary one.

