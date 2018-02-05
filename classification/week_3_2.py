
# coding: utf-8

# In[1]:


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


# In[2]:


loan_data = pd.read_csv('../data/lending-club-data.csv')


# In[3]:


loan_data.shape


# In[4]:


loan_data.head()


# In[5]:


loan_data['safe_loans'] = np.where(loan_data.bad_loans == 0, 1, -1)


# In[6]:


loan_data.safe_loans.value_counts()
loan_data = loan_data.drop(['bad_loans'], axis=1)


# In[7]:


features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'


# In[8]:


loan_data = loan_data[features + [target]]


# In[9]:


loan_data.head()


# In[10]:


def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod[target_column] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)


# In[11]:


def get_one_hot_encoding_for_categorical_variables(df):
    categorical_variables = []
    for feat_name, feat_type in zip(df.columns.values, df.dtypes):
        if feat_type == np.dtype('object'):
            categorical_variables.append(feat_name)
            df, _ = encode_target(df, feat_name)
    return df


# In[12]:


loans_df = get_one_hot_encoding_for_categorical_variables(loan_data)


# In[13]:


print(loan_data.shape, loans_df.shape)


# In[14]:


train_indexes = pd.read_json('../data/module-5-assignment-2-train-idx.json')
test_indexes = pd.read_json('../data/module-5-assignment-2-test-idx.json')


# In[15]:


print(train_indexes.shape, test_indexes.shape)


# In[16]:


train_data = loans_df.iloc[train_indexes[0].tolist()]
test_data = loans_df.iloc[test_indexes[0].tolist()]


# Steps to follow:
# 
# - Step 1: Calculate the number of safe loans and risky loans.
# - Step 2: Since we are assuming majority class prediction, all the data points that are not in the majority class are considered mistakes.
# - Step 3: Return the number of mistake

# In[17]:


def intermediate_node_num_mistakes(labels_in_node):
    '''
    :param: labels_in_node of type np.array or list
    '''
    if len(labels_in_node) == 0:
        return 0
    counts_dict = Counter(labels_in_node)
    if counts_dict[-1] > counts_dict[1]:
        return counts_dict[1]
    else:
        return counts_dict[-1]


# In[18]:


# Test case 1
example_labels = np.array([-1, -1, 1, 1, 1])
assert intermediate_node_num_mistakes(example_labels) == 2, 'Test 1 failed... try again!'

# Test case 2
example_labels2 = pd.Series([-1, -1, 1, 1, 1, 1, 1]).as_matrix()
assert intermediate_node_num_mistakes(example_labels2) == 2, 'Test 2 failed... try again!'
    
# Test case 3
example_labels = [-1, -1, -1, -1, -1, 1, 1]
assert intermediate_node_num_mistakes(example_labels) == 2, 'Test 3 failed... try again!'

