#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans


# In[2]:


wv = Word2Vec.load('word2vec.model').wv


# In[4]:


model = KMeans(n_clusters=2, max_iter=1000, random_state=True,n_init=50).fit(X=wv.vectors.astype('double'))


# In[5]:


wv.similar_by_vector(model.cluster_centers_[1], topn=10, restrict_vocab=None)


# In[7]:


positive_cluster_index = 1
positive_cluster_center = model.cluster_centers_[positive_cluster_index]
negative_cluster_center = model.cluster_centers_[1-positive_cluster_index]


# In[9]:


words = pd.DataFrame(wv.index_to_key)
words.columns = ['words']
words['vectors'] = words.words.apply(lambda x: wv[f'{x}'])
words['cluster'] = words.vectors.apply(lambda x: x[0])


# In[11]:


words['cluster_value'] = [1 if i==positive_cluster_index else -1 for i in words.cluster]
words['closeness_score'] = words.apply(lambda x: 1/(model.transform([x.vectors]).min()), axis=1)
words['sentiment_coefficient'] = words.closeness_score * words.cluster_value


# In[13]:


words.tail(10)


# In[15]:


words[['words', 'sentiment_coefficient']].to_csv('sentiment_dictionary.csv', index=False)


# In[ ]:




