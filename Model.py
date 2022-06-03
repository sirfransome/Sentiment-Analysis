#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
from gensim.models.phrases import Phrases, Phraser
from collections import defaultdict
import logging
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


# In[30]:


df = pd.read_excel("C:\\Users\\teniola.abiola\\Downloads\\Vbank Python\\Sentiment analysis\\Vbankng_clean_tweets.xlsx")


# In[31]:


df.head()


# In[32]:


df = df.dropna()


# In[33]:


df


# In[48]:


df.to_excel('Vbankng_clean_tweets.xlsx', index=False)


# In[34]:


#Phraser() takes a list of list of words as input
sent = [row.split() for row in df['Text']]


# In[35]:


sent


# In[36]:


phrases = Phrases(sent, min_count=30, progress_per=10000)


# In[37]:


biagram = Phraser(phrases)


# In[38]:


sentences = biagram[sent]


# In[39]:



word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
len(word_freq)


# In[42]:


#training the model
import multiprocessing
from gensim.models import Word2Vec
from time import time


# In[44]:


w2v_model = Word2Vec(min_count=3,
                     window=4,
                     vector_size=300,
                     sample=1e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=multiprocessing.cpu_count()-1)
#building vocabulary table
start = time()
w2v_model.build_vocab(sentences, progress_per=10000)

print('Time to build vocab: {} mins'.format(round((time() - start) / 60, 2)))


# In[ ]:





# In[45]:


#training the model
t = time()
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
w2v_model.init_sims(replace=True)


# In[46]:


w2v_model.save("word2vec.model")


# In[ ]:




