
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


# In[ ]:


loan_data = pd.read_csv('../data/lending-club-data.csv')


# In[ ]:


loan_data.shape


# In[ ]:


loan_data.head()


# In[ ]:


loan_data['safe_loans'] = np.where(loan_data.bad_loans == 0, 1, -1)


# In[ ]:


loan_data.safe_loans.value_counts()
loan_data = loan_data.drop(['bad_loans'], axis=1)


# In[ ]:


features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'


# In[ ]:


loan_data = loan_data[features + [target]]


# In[ ]:


loan_data.head()


# In[ ]:


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


# In[ ]:


def get_one_hot_encoding_for_categorical_variables(df):
    categorical_variables = []
    for feat_name, feat_type in zip(df.columns.values, df.dtypes):
        if feat_type == np.dtype('object'):
            categorical_variables.append(feat_name)
            df, _ = encode_target(df, feat_name)
    return df


# In[ ]:


loans_df = get_one_hot_encoding_for_categorical_variables(loan_data)


# In[ ]:


print(loan_data.shape, loans_df.shape)


# In[ ]:


train_indexes = pd.read_json('../data/module-5-assignment-2-train-idx.json')
test_indexes = pd.read_json('../data/module-5-assignment-2-test-idx.json')


# In[ ]:


print(train_indexes.shape, test_indexes.shape)


# In[ ]:


train_data = loans_df.iloc[train_indexes[0].tolist()]
test_data = loans_df.iloc[test_indexes[0].tolist()]


# Steps to follow:
# 
# - Step 1: Calculate the number of safe loans and risky loans.
# - Step 2: Since we are assuming majority class prediction, all the data points that are not in the majority class are considered mistakes.
# - Step 3: Return the number of mistake

# In[ ]:


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


# In[ ]:


# Test case 1
example_labels = np.array([-1, -1, 1, 1, 1])
assert intermediate_node_num_mistakes(example_labels) == 2, 'Test 1 failed... try again!'

# Test case 2
example_labels2 = pd.Series([-1, -1, 1, 1, 1, 1, 1]).as_matrix()
assert intermediate_node_num_mistakes(example_labels2) == 2, 'Test 2 failed... try again!'
    
# Test case 3
example_labels = [-1, -1, -1, -1, -1, 1, 1]
assert intermediate_node_num_mistakes(example_labels) == 2, 'Test 3 failed... try again!'


# In[2]:


def best_splitting_feature(data, features, target):    
    target_values = data[target]
    best_feature = None 
    best_error = 2     

    num_data_points = float(len(data))  
    
    for feature in features:
        
        left_split = data[data[feature] == 0]       
        right_split = data[data[feature] == 1] 
            
        # Calculate the number of misclassified examples in the left split.
        left_mistakes = intermediate_node_num_mistakes(left_split.target.as_matrix())
        right_mistakes = intermediate_node_num_mistakes(right_split.target.as_matrix())

        error = (left_mistakes + right_mistakes)/num_data_points

        if error < best_error:
            best_error = error
            best_feature = feature
    
    return best_feature 


# In[3]:


def create_leaf(target_values):    
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True 
           }
   
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])    

    if num_ones > num_minus_ones:
        leaf['prediction'] = 1  
    else:
        leaf['prediction'] =  -1
        
    return leaf

