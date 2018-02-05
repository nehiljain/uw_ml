
# coding: utf-8

# In[37]:


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


# In[38]:


loan_data = pd.read_csv('../data/lending-club-data.csv')


# In[39]:


loan_data.shape


# In[40]:


loan_data.head()


# In[41]:


loan_data['safe_loans'] = np.where(loan_data.bad_loans == 0, 1, -1)


# In[42]:


loan_data.safe_loans.value_counts()
loan_data = loan_data.drop(['bad_loans'], axis=1)


# In[43]:


features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'


# In[44]:


loan_data = loan_data[features + [target]]


# In[45]:


loan_data.head()


# In[46]:


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


# In[47]:


def get_one_hot_encoding_for_categorical_variables(df):
    categorical_variables = []
    for feat_name, feat_type in zip(df.columns.values, df.dtypes):
        if feat_type == np.dtype('object'):
            categorical_variables.append(feat_name)
            df, _ = encode_target(df, feat_name)
    return df


# In[48]:


loans_df = pd.get_dummies(loan_data, columns=features)


# In[51]:


features = list(loans_df.columns.values)
features.remove(target)
features


# In[52]:


print(loan_data.shape, loans_df.shape)


# In[53]:


train_indexes = pd.read_json('../data/module-5-assignment-2-train-idx.json')
test_indexes = pd.read_json('../data/module-5-assignment-2-test-idx.json')


# In[54]:


print(train_indexes.shape, test_indexes.shape)


# In[55]:


train_data = loans_df.iloc[train_indexes[0].tolist()]
test_data = loans_df.iloc[test_indexes[0].tolist()]


# Steps to follow:
# 
# - Step 1: Calculate the number of safe loans and risky loans.
# - Step 2: Since we are assuming majority class prediction, all the data points that are not in the majority class are considered mistakes.
# - Step 3: Return the number of mistake

# In[56]:


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


# In[57]:


# Test case 1
example_labels = np.array([-1, -1, 1, 1, 1])
assert intermediate_node_num_mistakes(example_labels) == 2, 'Test 1 failed... try again!'

# Test case 2
example_labels2 = pd.Series([-1, -1, 1, 1, 1, 1, 1]).as_matrix()
assert intermediate_node_num_mistakes(example_labels2) == 2, 'Test 2 failed... try again!'
    
# Test case 3
example_labels = [-1, -1, -1, -1, -1, 1, 1]
assert intermediate_node_num_mistakes(example_labels) == 2, 'Test 3 failed... try again!'


# In[58]:


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


# In[59]:


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


# In[60]:


def decision_tree_create(data, features, target, current_depth = 0, max_depth = 10):
    remaining_features = features[:]
    target_values = data[target]
    print("--------------------------------------------------------------------")
    print("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))

#     print("remaining_features {}".format(remaining_features))
    if intermediate_node_num_mistakes(target_values.as_matrix()) == 0:  
        print("Stopping condition 1 reached.")
        return create_leaf(target_values)
    if not remaining_features:
        print("Stopping condition 2")
        return create_leaf(target_values)
    if current_depth >= max_depth:
        print("Reached max_depth = {}".format(current_depth))
        return create_leaf(target_values)
    
    splitting_feature = best_splitting_feature(data, remaining_features, target)
    # because we know all features are binary! (I got stuck here for a sec)
    left_split_data = data[data[splitting_feature] == 0]
    right_split_data = data[data[splitting_feature] == 1]
    remaining_features.remove(splitting_feature)
    print("Split on feature %s. (%s, %s)" % (                      splitting_feature, len(left_split_data), len(right_split_data)))

    if len(left_split_data) == len(data):
        print("Creating leaf node.")
        return create_leaf(left_split_data[target])
    if len(right_split_data) == len(data):
        print("Creating leaf node.")
        return create_leaf(right_split_data[target])
    
    
    left_tree = decision_tree_create(left_split_data, remaining_features, 
                                     target, current_depth + 1, max_depth)        
    ## YOUR CODE HERE
    right_tree = decision_tree_create(right_split_data, remaining_features, 
                                      target, current_depth + 1, max_depth)                   
    
    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}
              


# In[61]:


def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])


# In[62]:


small_data_decision_tree = decision_tree_create(train_data, features, 'safe_loans', max_depth = 3)
if count_nodes(small_data_decision_tree) == 13:
    print('Test passed!')
else:
    print('Test failed... try again!')
    print('Number of nodes found                :', count_nodes(small_data_decision_tree))
    print('Number of nodes that should be there : 13')


# In[63]:


my_decision_tree = decision_tree_create(train_data, features, 'safe_loans', max_depth= 6)


# In[64]:


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


# In[69]:


test_data.iloc[0]


# In[70]:


print(test_data.iloc[0])
print('Predicted class: %s ' % classify(my_decision_tree, test_data.iloc[0]))


# In[72]:


classify(my_decision_tree, test_data.iloc[0], annotate=True)


# In[88]:


def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x), axis=1).as_matrix()
    
    num_errors = sum(np.where(prediction != data[target], 1, 0))
#     fp = sum((prediction == 1) & (data[target] == -1))
#     fn = sum((prediction == -1) & (data[target] == 1))
#     tp = sum((prediction == 1) & (data[target] == 1))
#     tn = sum((prediction == -1) & (data[target] == -1))
#     print(fp, fn, tp, tn)
#     recall = float(tp)/(tp + fn)
#     precision = float(tp)/(tp + fp)
#     f1 = 2 * (precision * recall) / (precision + recall)
    error = float(num_errors)/len(data)
    return error


# In[89]:


evaluate_classification_error(my_decision_tree, test_data)


# In[ ]:


def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print("(leaf, label: %s)" % tree['prediction'])
        return None
    split_feature, split_value = split_name.split('.')
    print('                       %s' % name)
    print('         |---------------|----------------|')
    print('         |                                |'
    print('         |                                |'
    print'         |                                |'
    print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                         (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))

