
# coding: utf-8

# In[36]:


import pandas as pd
df = pd.read_csv("D:\DA_ASSIGNMENT\PROJECT\project.csv",parse_dates =["date"],index_col ="date")


# In[37]:


#dropping the columns which are unnecessary.
df1=df.drop(columns=['id','month','hour'])


# In[42]:


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


# In[49]:


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

