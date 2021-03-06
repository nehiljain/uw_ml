
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import json
import sklearn
import sklearn.ensemble
from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


loans = pd.read_csv('../../uw_ml/data/lending-club-data.csv')


# In[ ]:


# safe_loans =  1 => safe
# safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop('bad_loans', axis = 1)


# In[ ]:


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


# In[ ]:


loans = loans[[target] + features].dropna()


# In[ ]:


loans = pd.get_dummies(loans)


# In[ ]:



with open('../../uw_ml/data/module-8-assignment-1-train-idx.json', 'r') as f: # Reads the list of most frequent words
    train_idx = json.load(f)
with open('../../uw_ml/data/module-8-assignment-1-validation-idx.json', 'r') as f1: # Reads the list of most frequent words
    validation_idx = json.load(f1)


# In[ ]:


train_data = loans.iloc[train_idx]
validation_data = loans.iloc[validation_idx]


# In[ ]:


validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
sample_validation_data


# In[ ]:


model_5 = GradientBoostingClassifier(n_estimators=5, 
                                          max_depth=6)


# In[ ]:


X = train_data.drop('safe_loans',1)


# In[ ]:


np.any(np.isnan(X))


# In[ ]:


model_5.fit(X, train_data['safe_loans'])


# In[ ]:


model_5.predict(sample_validation_data.drop('safe_loans',1))


# In[ ]:


model_5.predict_proba(sample_validation_data.drop('safe_loans',1))


# In[ ]:


model_5.score(validation_data.drop('safe_loans',1), validation_data['safe_loans'])


# In[ ]:


predict_safeloans = model_5.predict(validation_data.drop('safe_loans',1))


# In[ ]:


predict_safeloans


# In[ ]:


fp = sum(predict_safeloans > validation_data['safe_loans'])
fp


# In[ ]:


# false negative
fn = sum(predict_safeloans < validation_data['safe_loans'])
fn


# In[ ]:


validation_data['predictions'] = model_5.predict_proba(validation_data.drop('safe_loans',1))[:,1]


# In[ ]:


assert sum((validation_data.predictions > 1) | (validation_data.predictions < 0)) == 0


# In[ ]:


validation_data[['grade_A','grade_B','grade_C','grade_D','predictions']].sort_values('predictions', ascending = False).head(5)


# In[ ]:


mistake_cost = (10000 * fn) + (20000 * fp)
mistake_cost


# In[ ]:


models = {}
val_prediction_data = validation_data.drop('safe_loans', 1)
val_label_data = validation_data['safe_loans']
for estimator in [5, 10,50,100,200,500]:
    model = GradientBoostingClassifier(n_estimators=10,
                                     max_depth=6)
    model.fit(X, train_data['safe_loans'])
    predictions = model.predict(val_prediction_data)
    fn = sum(predictions < val_label_data)
    fp = sum(predictions > val_label_data)
    models['model_' + str(estimator)] = {
        'model': model,
        'fn': sum,
        'fp': ,
        'sample_predictions': model.predict(sample_validation_data.drop('safe_loans',1)),
        ''
    print()


# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


old_bid = pd.read_csv('/Users/nehiljain/Downloads/snaptravel_trivago_bids.csv', sep=';')


# In[3]:


new_bid = pd.read_csv('/Users/nehiljain/Downloads/snaptravel_trivago_bids_2018-03-07-21-00.csv', sep=';')


# In[4]:


new_bid.head()


# In[ ]:


sum(new_bid.US < old_bid.US)


# In[ ]:


sns.distplot(new_bid.US, hist=False, rug=True)
sns.distplot(old_bid.US, hist=False, rug=True)
sns.plt.show()

