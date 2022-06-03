#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re


# In[54]:


tweets = pd.read_excel("C:\\Users\\teniola.abiola\\Downloads\\Vbank Python\\Sentiment analysis\\Vbankng_tweets.xlsx")


# In[55]:


tweets


# In[56]:


tweets = tweets[['Username', 'Text']]
tweets


# In[57]:


tweets.describe()


# In[58]:


tweets['Text'][0]


# In[59]:


sns.heatmap(tweets.isnull(), yticklabels=False)
plt.show()


# In[60]:


len(tweets)


# In[61]:


length = list()
for i in range(len(tweets)):
    length.append(len(tweets.iloc[i, 1]))


# In[62]:


tweets['Length'] = length


# In[63]:


tweets


# In[64]:


tweets =tweets.drop('Username', axis=1)
tweets


# In[65]:


plt.hist(tweets['Length'], bins = 100)
plt.show()


# In[66]:


tweets.describe()


# In[67]:


tweets[tweets['Length']== min(tweets['Length'])]['Text'].iloc[0]


# In[68]:


tweets[tweets['Length']== max(tweets['Length'])]['Text'].iloc[0]


# In[69]:


from wordcloud import WordCloud
sentences = tweets['Text'].tolist()
com_sen = " ".join(sentences)


# In[70]:


plt.imshow(WordCloud().generate(com_sen))
plt.show()


# In[71]:


tweets['Text'] = tweets['Text'].apply(lambda x: x.replace("vbankng", ""))
tweets['Text'] = tweets['Text'].apply(lambda x: x.replace("DONJAZZY", ""))
tweets['Text'] = tweets['Text'].apply(lambda x: x.replace("https", ""))


# In[72]:


tweets


# In[73]:


import string
string.punctuation
import nltk


# In[74]:


nltk.download('stopwords')


# In[75]:


from nltk.corpus import stopwords
st = stopwords.words('english')
st.append('and')


# In[ ]:





# In[76]:


def message_cleaning(message):
    test_punc_removed = [char for char in message if char not in string.punctuation]
    test_punc_removed = ''.join(test_punc_removed)
    test_punc_st_removed = []
    for char in test_punc_removed.split():
        if char.lower() not in st:
            test_punc_st_removed.append(char)
    test_punc_st_removed = " ".join(test_punc_st_removed)  
    return test_punc_st_removed


# In[77]:


clean_data = tweets['Text'].apply(message_cleaning)


# In[79]:


clean_data


# In[88]:


def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)


# In[95]:


clean_data = clean_data.apply(remove_emojis)


# In[96]:


clean_data


# In[101]:


clean_data = clean_data[clean_data.notnull()]


# In[103]:


clean_data = clean_data.dropna()


# In[105]:


clean_data = clean_data.reset_index()


# In[98]:


clean_data.to_excel("C:\\Users\\teniola.abiola\\Downloads\\Vbank Python\\Sentiment analysis\\Vbankng_clean_tweets.xlsx", index=False)


# In[ ]:




