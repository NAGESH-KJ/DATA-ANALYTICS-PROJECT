#asr_zone represents Accessor's zone code for nearest parcel.
# a bar plot to plot number of crimes VS diff asr_zone
import pandas as pd
#Importing pandas
import matplotlib.pyplot as plt
#importing mathplot 
import numpy as np
#importing numpy

def plot_bar_x():
    index = np.arange(len(list1))
    plt.bar(index,numOfCrimesInLocality)
    #giving the xlabel
    plt.xlabel('', fontsize=7)
    #giving the ylabel
    plt.ylabel('No of Crimes', fontsize=7)
    plt.xticks(index, list1, fontsize=7, rotation=30)
    #title of the bar plot
    plt.title('Number of crimes in respective locality')
    #plotting the bar plot.
    plt.show()
    


df=pd.read_csv("incidents-100k.csv")
#import the dataset that is required
l=df.asr_zone.unique()
l.sort()
#All the unique values of the asr_zone
list1=[]
# assigning appropriate types to the values of the . azr_zone

for j in l:
    if(j == -1):
        list1.append("Unset")
    elif(j==0):
        list1.append("Unzoned")
    elif(j==1):
        list1.append("Single Family Resident")
    elif(j==2):
        list1.append("Minor Multiple")
    elif(j==3):
        list1.append("Restricted Multiple")
    elif(j==4):
        list1.append("Multiple Resident")
    elif(j==5):
        list1.append("Restricted commerical")
    elif(j==6):
        list1.append("Commericial")
    elif(j==7):
        list1.append("Industrial")
    elif(j==8):
        list1.append("Agricultural")
    elif(j==9):
        list1.append("Special")

#A list to store number of crimes for repective types
numOfCrimesInLocality=[]
# to find the number of crimes for each type
for i in l:
    #finding the subset of the dataset for each asr_zone.
    subDf= df.loc[df.asr_zone==i,]
    numOfCrimesInLocality.append(subDf.shape[0])
#printing the number of crimes for each asr_zone
print(numOfCrimesInLocality)
#o=plotting the bar graph
plot_bar_x()
#So we observe that Most crimes occur in Commercial asr_zone.

#Subset of the dataset where crimes were commited in Single Family Resident zone.
SingleFamilydf=df.loc[df.asr_zone==1,]
#An empty dictionary
d={}
for i in SingleFamilydf.type:
    if(i not in d):
        d[i]=1
    else:
        d[i]=d[i]+1

max1=0
crime=""
for key in d:
    if(d[key] >max1):
        max1=d[key]
        crime=key
#the most type of crime that was committed was of the type
print("Type of crime most occurring in commercial zone is ",crime)
print(max1)


#Subset of the dataset where crimes were commited in Commercial
Comdf=df.loc[df.asr_zone==6,]
#An empty dictionary
d1={}
for i in Comdf.type:
    if(i not in d1):
        d1[i]=1
    else:
        d1[i]=d1[i]+1

max2=0
crime1=""
for key in d:
    if(d1[key] >max2):
        max2=d1[key]
        crime1=key
#the most type of crime that was committed was of the type
print("Type of crime most occurring in commercial zone is ",crime1)
print(max2)
