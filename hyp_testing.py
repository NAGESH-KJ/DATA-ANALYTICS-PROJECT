# A z test is applied since a moderate sized  sample of the dataset is taken at random
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import scipy.stats as stats 
from scipy.special import ndtr as ndtr
#Hypothesis testing 
# H0: Number of crimes in day > = crimes in night
# H1: Number of crimes in day < number of crimes in night.
df=pd.read_csv("incidents-100k.csv")
#A list for number of crimes in days
mean=df.is_night.mean()
#This is to know The mean  and standared deviation of the sampling distribution of sample means.
sigma=df.is_night.std(ddof=0)
# Take a random sample.
randomSample = df.sample(n=50)
x_bar=randomSample.is_night.mean()
#Intial step is we assume NUll hypothsis is true , so  number of crimes in day = number of crimes in night.
# Now we take the area in disagreement with the null hypothesis of the newly generated sample.
z_critical = 1.96 #Standard confidence level
N = 30
SE = sigma/np.sqrt(N)
z_stat = (x_bar - mean)/SE
pval = 1- ndtr(z_stat)
# applying the standard formulaes
# if the pvalues is less than 0.05 then we reject the null hypothesis
if pval < 0.05:
  print("we reject null hypothesis")
else:
  print("we fail to reject null hypothesis")
#So we see that we fail to reject that more crimes take place in day than in night
#More number crimes were occuring in 3 and 4 am ,due this reason we are getting the conclusion that we cant
#infer that crimes in night is greater than crimes in day time.
