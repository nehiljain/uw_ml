
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np


# In[39]:


loans = pd.read_csv('../../uw_ml/data/lending-club-data.csv')


# In[40]:


# safe_loans =  1 => safe
# safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop('bad_loans', axis = 1)


# In[41]:


features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]


# In[42]:


target = 'safe_loans'


# In[43]:


loans = loans[[target] + features]


# In[44]:


loans = pd.get_dummies(loans)


# In[45]:


with open('../../uw_ml/data/module-8-assignment-2-train-idx.json', 'r') as f: # Reads the list of most frequent words
    train_idx = json.load(f)
with open('../../uw_ml/data/module-8-assignment-2-test-idx.json', 'r') as f1: # Reads the list of most frequent words
    test_idx = json.load(f1)


# In[48]:


train_data = loans.iloc[train_idx].reset_index()
test_data = loans.iloc[test_idx].reset_index()


# In[49]:


train_data = train_data.drop('index', 1)
test_data = test_data.drop('index',1)


# In[50]:


def intermediate_node_weighted_mistakes(labels_in_node, data_weights):
    # Sum the weights of all entries with label +1
    total_weight_positive = sum(data_weights[labels_in_node == +1])
    
    # Weight of mistakes for predicting all -1's is equal to the sum above
    ### YOUR CODE HERE
    weighted_mistakes_all_negative = total_weight_positive
    
    # Sum the weights of all entries with label -1
    ### YOUR CODE HERE
    total_weight_negative = sum(data_weights[labels_in_node == -1])
    
    # Weight of mistakes for predicting all +1's is equal to the sum above
    ### YOUR CODE HERE
    weighted_mistakes_all_positive = total_weight_negative
    
    # Return the tuple (weight, class_label) representing the lower of the two weights
    #    class_label should be an integer of value +1 or -1.
    # If the two weights are identical, return (weighted_mistakes_all_positive,+1)
    ### YOUR CODE HERE
    if weighted_mistakes_all_positive > weighted_mistakes_all_negative:
        return (weighted_mistakes_all_negative, -1)
    else:
        return (weighted_mistakes_all_positive, +1)
    

