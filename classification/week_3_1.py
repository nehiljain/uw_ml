
# coding: utf-8

# ## Assignment 1 Week 3
# 
# Decision Tree

# In[45]:


import sys, os
from datetime import datetime, timedelta,date
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('classic')
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import json
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


# In[27]:


loan_data = pd.read_csv('../data/lending-club-data.csv')


# In[28]:


loan_data.shape


# In[29]:


loan_data.head()


# In[30]:


loan_data['safe_loans'] = np.where(loan_data.bad_loans == 0, 1, -1)


# In[31]:


loan_data.safe_loans.value_counts()


# In[32]:


features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loan_data = loan_data[features + [target]]


# In[33]:


train_indexes = pd.read_json('../data/module-5-assignment-1-train-idx.json')
val_indexes = pd.read_json('../data/module-5-assignment-1-validation-idx.json')


# In[34]:


train_data = loan_data.iloc[train_indexes[0].tolist()]
val_data = loan_data.iloc[val_indexes[0].tolist()]


# In[35]:


safe_loans_raw = train_data[train_data[target] == +1]
risky_loans_raw = train_data[train_data[target] == -1]
print("Number of safe loans  : %s" % len(safe_loans_raw))
print("Number of risky loans : %s" % len(risky_loans_raw))


# In[36]:


# Since there are fewer risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
print(percentage)


# In[37]:


risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(len(risky_loans_raw), random_state=1)


# In[38]:


loans_df = pd.concat([risky_loans, safe_loans])


# In[39]:


loans_df.safe_loans.value_counts()


# In[40]:


loans_df.columns.values


# In[41]:


categorical_variables = []
for feat_name, feat_type in zip(loans_df.columns.values, loans_df.dtypes):
    if feat_type == np.dtype('object'):
        categorical_variables.append(feat_name)


# In[42]:


categorical_variables


# In[43]:


loans_df = pd.get_dummies(loans_df, columns=categorical_variables)


# In[44]:


loans_df.head()

