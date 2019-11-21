# this might take a minute or two to run.

#importing the reqiured modules
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
  
# Reads the csv file  
df=pd.read_csv("incidents-100k.csv")
# A random Sample is taken because , the dataset contains 100k rows and will reqiure some time  to run.
df = df.sample(n=10000)
comment_words = ' '
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in df.desc: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
          
    for words in tokens: 
            comment_words = comment_words + words + ' '
#generating the word cloud.
        
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                collocations = False,
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
#A wordcloud analysis was done on desc column  of the dataset , we find that theft and vehicle were the most weighted ones, which shows that most of the crimes  that occurred in the city were related to thefts or  offences that involved  vehicles.
#The word cloud also shows up alcohol and drugs violations.
