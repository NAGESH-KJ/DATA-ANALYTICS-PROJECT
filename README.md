# DATA-ANALYTICS-PROJECT
CRIME DATA ANALYSIS

Link for the dataset : https://data.sandiegodata.org/dataset/san-diego-region-crime-incidents-2007-2013/

For the project, we have used the 'incidents-100k.csv' file from the above link.


As preprocessing part of the dataset , we have implemented Dimensionality Reduction. We have dropped certain columns since someof them are not useful for analysis and some are redundant. ID and city are useless variables.Hour is a redundant variable since it can be derived from time column.

CrimesInSanDiego-contains certain graphs as a part of data visualization. R , Rmd and html files for the same is uploaded. We have used different types of plots i.e, bar chart and pie chart for visualization.

CrimeVisualizationForDifferentasr_zone - To determine in which zone has maximum crimes and to determine the type of crime that is more prevalant in that zone.

hyp_testing - To determine whether there are more crime activities in the night or daytime using hypothesis testing.

word_cloud - To analysize the density of different types of crimes based on the description of the crime using wordcloud.

timeseries1 - This is to determine the trend in crime occurrance pattern near the lampost and to check whether it is stationary or not.ADF test is done in this file.

NaiveBayesModel - In this file there is an attempt to classify the description into different type of crimes using naive bayes classifier. There is feature extraction also  done to identify most correlated unigrams and bigrams.

k_means - It is an attempt to cluster the crimes based on their locations.

further_analysis - Dataset is resampled on monthly basis and then trends in number of crimes is analysed. Also, we try to understand the distribution of crimes in different neighbourhoods. Also, we visualize  the time intervals in which rate of time occurance is more.

All the above mentioned files except the R have been combined into a single file for ease of evaluation. The filename is 'DataMiners_final.ipynb' . DataMiners.py and DataMiners.html files have also been uploaded. 
