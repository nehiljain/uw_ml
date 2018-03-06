
# coding: utf-8

# In[14]:


import sys, os
from collections import Counter
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
from sklearn.tree import DecisionTreeClassifier, export_graphviz


# In[15]:


loan_data = pd.read_csv('../data/lending-club-data.csv')


# In[16]:


loan_data.shape


# In[17]:


loan_data.columns.values


# In[18]:


loan_data['safe_loans'] = loan_data['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loan_data = loan_data.drop(columns='bad_loans')


# In[19]:


target = 'safe_loans'
features = ['grade',                     # grade of the loan (categorical)
            'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'payment_inc_ratio',         # ratio of the monthly payment to income
            'delinq_2yrs',               # number of delinquincies
             'delinq_2yrs_zero',          # no delinquincies in last 2 years
            'inq_last_6mths',            # number of creditor inquiries in last 6 months
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'open_acc',                  # number of open credit accounts
            'pub_rec',                   # number of derogatory public records
            'pub_rec_zero',              # no derogatory public records
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            'int_rate',                  # interest rate of the loan
            'total_rec_int',             # interest received to date
            'annual_inc',                # annual income of borrower
            'funded_amnt',               # amount committed to the loan
            'funded_amnt_inv',           # amount committed by investors for the loan
            'installment',               # monthly payment owed by the borrower
           ]


# In[20]:


loan_data = loan_data[[target] + features]


# In[21]:


loan_data.shape


# In[22]:


loan_data = loan_data.dropna()


# In[23]:


loan_data.shape


# In[24]:


train_indexes = pd.read_json('../data/module-8-assignment-1-train-idx.json')
val_indexes = pd.read_json('../data/module-8-assignment-1-validation-idx.json')


# In[30]:


train_data = loan_data.loc[train_indexes[0].tolist()]
validation_data = loan_data.loc[val_indexes[0].tolist()]

