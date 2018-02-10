
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


loans_df = pd.get_dummies(loan_data, columns=features)


# In[11]:


features = list(loans_df.columns.values)
features.remove(target)
features


# In[12]:


print(loan_data.shape, loans_df.shape)


# In[13]:


train_indexes = pd.read_json('../data/module-5-assignment-2-train-idx.json')
test_indexes = pd.read_json('../data/module-5-assignment-2-test-idx.json')


# In[14]:


print(train_indexes.shape, test_indexes.shape)


# In[15]:


train_data = loans_df.iloc[train_indexes[0].tolist()]
test_data = loans_df.iloc[test_indexes[0].tolist()]


# In[16]:


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


# In[17]:


def best_splitting_feature(data, features, target):    
    target_values = data[target]
    best_feature = None 
    best_error = 2     

    num_data_points = float(len(data))  
    
    for feature in features:
        
        left_split = data[data[feature] == 0]       
        right_split = data[data[feature] == 1] 
            
        # Calculate the number of misclassified examples in the left split.
        left_mistakes = intermediate_node_num_mistakes(left_split[target].as_matrix())
        right_mistakes = intermediate_node_num_mistakes(right_split[target].as_matrix())

        error = (left_mistakes + right_mistakes)/num_data_points

        if error < best_error:
            best_error = error
            best_feature = feature
    
    return best_feature 


# In[18]:


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


# In[19]:


def classify(tree, x, annotate = False):
    if tree['is_leaf']:
        if annotate:
             print("At leaf, predicting %s" % tree['prediction'])
        return tree['prediction']
    else:
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
             print("Split on %s = %s" % (tree['splitting_feature'], split_feature_value))
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['left'], x, annotate)


# In[20]:


def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x), axis=1).as_matrix()
    
    num_errors = sum(np.where(prediction != data[target], 1, 0))

    error = float(num_errors)/len(data)
    return error


# In[21]:


def reached_minimum_node_size(data, min_node_size):
    return data.shape[0] <= min_node_size


# In[22]:


def error_reduction(error_before_split, error_after_split):
    return error_before_split - error_after_split


# In[23]:


def decision_tree_create(data, features, target, 
                         current_depth = 0, 
                         max_depth = 10, 
                         min_node_size=1, 
                         min_error_reduction=0.0):
    remaining_features = features[:]
    target_values = data[target]
    print("--------------------------------------------------------------------")
    print("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))


    if intermediate_node_num_mistakes(target_values.as_matrix()) == 0:  
        print("Stopping condition 1 reached.")
        return create_leaf(target_values)
    if not remaining_features:
        print("Stopping condition 2")
        return create_leaf(target_values)
    if current_depth >= max_depth:
        print("Reached max_depth = {}".format(current_depth))
        return create_leaf(target_values)
    if reached_minimum_node_size(data, min_node_size):
        print("Reached early stopping condition2 min node size")
        return create_leaf(data[target])
    
    splitting_feature = best_splitting_feature(data, remaining_features, target)
    # because we know all features are binary! (I got stuck here for a sec)
    left_split_data = data[data[splitting_feature] == 0]
    right_split_data = data[data[splitting_feature] == 1]
    remaining_features.remove(splitting_feature)
    print("Split on feature %s. (%s, %s)" % (                      splitting_feature, len(left_split_data), len(right_split_data)))

    error_before_split = intermediate_node_num_mistakes(target_values) / float(len(data))
    left_mistakes = intermediate_node_num_mistakes(left_split_data[target].as_matrix())
    right_mistakes = intermediate_node_num_mistakes(right_split_data[target].as_matrix())
    error_after_split = (left_mistakes + right_mistakes) / float(len(data))
    
    if error_reduction(error_before_split, error_after_split) <= min_error_reduction:
        print("Early stopping condition min reduction")
        return create_leaf(target_values)
    
    
    left_tree = decision_tree_create(left_split_data, remaining_features, 
                                     target, current_depth + 1, max_depth)        
    right_tree = decision_tree_create(right_split_data, remaining_features, 
                                      target, current_depth + 1, max_depth)                   
    
    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}
              

