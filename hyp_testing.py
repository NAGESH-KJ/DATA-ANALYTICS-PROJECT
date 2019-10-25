
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
#Hypothesis testing 
# H0: Number of crimes in day > = crimes in night
# H1: Number of crimes in day < number of crimes in night.
df=pd.read_csv("incidents-100k.csv")
# taking a random sample of the data
randomSample = df.sample(n=60)
l=df.is_night.unique()
list1=[]
# count the number of crimes taking plce during day and night respectively 
for i in l:
    subDf= randomSample.loc[df.is_night==i,]
    list1.append(subDf.shape[0])
day=list1[0]
night=list1[1]
print(day)
print(night)
#finding the p value
ttest,pval = ttest_ind(day,night)
# if the pvalues is less than 0.05 then we reject the null hypothesis
if pval <0.05:
  print("we reject null hypothesis")
else:
  print("we fail to reject null hypothesis")
#So we see that we fail to reject that more crimes take place in day than in night

