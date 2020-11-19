#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import codecs
import pandas as pd
from tqdm.notebook import tqdm


# In[6]:


f_list = glob.glob('data/book-dummy/text_*')


# In[7]:


documents = []
for f in f_list:
    with codecs.open(f, mode='r', encoding='utf-8') as rf:
        docs = rf.read()
    documents.append(docs.split('\n'))


# In[8]:


len(documents)


# In[9]:


train_docs = documents[:8]
valid_docs = documents[8]
test_docs = documents[9]


# In[10]:


train_docs = [[sent.lstrip("\'").rstrip("\'").replace('"', '').replace("`", '') for sent in doc] for doc in tqdm(train_docs)]


# In[11]:


paragraphs = []
paragraph = []
for doc in tqdm(train_docs):
    for idx, l in enumerate(doc):
        if (len(paragraph) + 1) % 20 == 0:
            paragraph.append(l)
            paragraphs.append(paragraph)
            paragraph = []
        else:
            paragraph.append(l)


# In[13]:


train_meta = []
train_seg_list = []
for idx, para in enumerate(paragraphs):
    para_id = 'book-{:06d}'.format(idx)
    split = 'train'
    train_meta.append([para_id, split])
    for k, sent in enumerate(para):
        seg_id = para_id + '-{:04d}'.format(k)
        seg = sent
        train_seg_list.append([seg_id, seg])


# In[15]:


train_meta_df = pd.DataFrame(train_meta)


# In[16]:


train_seg_df = pd.DataFrame(train_seg_list)


# In[17]:


train_meta_df.columns = ['paragraph_id', 'type']


# In[18]:


train_seg_df.columns = ['UID', 'SEG']


# In[26]:


train_seg_df


# In[27]:


valid_paragraphs = []
valid_paragraph = []
for doc in tqdm([valid_docs]):
    for idx, l in enumerate(doc):
        if (len(valid_paragraph) + 1) % 20 == 0:
            valid_paragraph.append(l)
            valid_paragraphs.append(valid_paragraph)
            valid_paragraph = []
        else:
            valid_paragraph.append(l)


# In[28]:


valid_meta = []
valid_seg_list = []
for idx, para in enumerate(valid_paragraphs):
    para_id = 'book-{:06d}'.format(idx+len(train_meta_df))
    split = 'valid'
    valid_meta.append([para_id, split])
    for k, sent in enumerate(para):
        seg_id = para_id + '-{:04d}'.format(k)
        seg = sent
        valid_seg_list.append([seg_id, seg])


# In[29]:


for item in valid_meta:
    train_meta_df.loc[len(train_meta_df)] = item


# In[30]:


for item in tqdm(valid_seg_list):
    train_seg_df.loc[len(train_seg_df)] = item


# In[31]:


test_paragraphs = []
test_paragraph = []
for doc in tqdm([test_docs]):
    for idx, l in enumerate(doc):
        if (len(test_paragraph) + 1) % 20 == 0:
            test_paragraph.append(l)
            test_paragraphs.append(test_paragraph)
            test_paragraph = []
        else:
            test_paragraph.append(l)


# In[32]:


test_meta = []
test_seg_list = []
for idx, para in enumerate(test_paragraphs):
    para_id = 'book-{:06d}'.format(idx+len(train_meta_df))
    split = 'test'
    test_meta.append([para_id, split])
    for k, sent in enumerate(para):
        seg_id = para_id + '-{:04d}'.format(k)
        seg = sent
        test_seg_list.append([seg_id, seg])


# In[33]:


for item in test_meta:
    train_meta_df.loc[len(train_meta_df)] = item


# In[34]:


for item in tqdm(test_seg_list):
    train_seg_df.loc[len(train_seg_df)] = item


# In[35]:


train_seg_df


# In[36]:


train_meta_df


# In[37]:


train_meta_df.to_csv("data/book/book_dummy_meta.csv", index=None)


# In[38]:


train_seg_df.to_csv("data/book/book_dummy_main.csv", index=None)


# In[ ]:




