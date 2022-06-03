#!/usr/bin/env python
# coding: utf-8

# In[4]:


# importing libraries and packages
import snscrape.modules.twitter as sntwitter
import pandas
import time
import pandas as pd



# Creating list to append tweet data 
tweet_list = []

# Using TwitterSearchScraper to scrape data and append tweets to list
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('vbankng since:2020-01-01 until:2022-05-24').get_items()): 
    if i>16000: #number of tweets you want to scrape
        break
    tweet_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username]) #declare the attributes to be returned

# Creating a dataframe from the tweets list above 
df = pd.DataFrame(ManUtd_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])


# In[5]:


df


# In[10]:


df['Datetime'] = df['Datetime'].astype('str')
for x in df['Datetime']:
    df['Datetime']= x[:19]
    


# In[11]:


df


# In[13]:


df['Datetime'] = pd.to_datetime(df['Datetime'])


# In[14]:


df.dtypes


# In[15]:


df.to_excel('Vbankng_tweets.xlsx', index=False)


# In[ ]:




