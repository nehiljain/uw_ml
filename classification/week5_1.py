
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
import json
import sklearn
import sklearn.ensemble
from sklearn.ensemble import GradientBoostingClassifier


# In[46]:


loans = pd.read_csv('../../uw_ml/data/lending-club-data.csv')


# In[47]:


# safe_loans =  1 => safe
# safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop('bad_loans', axis = 1)


# In[48]:


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


# In[49]:


loans = loans[[target] + features].dropna()


# In[50]:


loans = pd.get_dummies(loans)


# In[51]:



with open('../../uw_ml/data/module-8-assignment-1-train-idx.json', 'r') as f: # Reads the list of most frequent words
    train_idx = json.load(f)
with open('../../uw_ml/data/module-8-assignment-1-validation-idx.json', 'r') as f1: # Reads the list of most frequent words
    validation_idx = json.load(f1)


# In[52]:


train_data = loans.iloc[train_idx]
validation_data = loans.iloc[validation_idx]


# In[53]:


validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
sample_validation_data


# In[54]:


sample_model = GradientBoostingClassifier(n_estimators=5, max_depth=6)


# In[55]:


X = train_data.drop('safe_loans',1)


# In[56]:


np.any(np.isnan(X))


# In[57]:


sample_model.fit(X, train_data['safe_loans'])


# In[58]:


sample_model.predict(sample_validation_data.drop('safe_loans',1))


# In[59]:


sample_model.predict_proba(sample_validation_data.drop('safe_loans',1))


# In[60]:


sample_model.score(validation_data.drop('safe_loans',1), validation_data['safe_loans'])


# In[61]:


predict_safeloans = sample_model.predict(validation_data.drop('safe_loans',1))


# In[62]:


predict_safeloans


# In[63]:


sum(predict_safeloans > validation_data['safe_loans'])


# In[64]:


# false negative
sum(predict_safeloans < validation_data['safe_loans'])


# In[65]:


validation_data['predictions'] = sample_model.predict_proba(validation_data.drop('safe_loans',1))[:,1]


# In[67]:


validation_data[['grade_A','grade_B','grade_C','grade_D','predictions']].sort_values('predictions', ascending = False).head(5)

