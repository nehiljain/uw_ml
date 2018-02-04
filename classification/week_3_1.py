
# coding: utf-8

# ## Assignment 1 Week 3
# 
# Decision Tree

# In[1]:


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


# In[7]:


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


# In[8]:


train_indexes = pd.read_json('../data/module-5-assignment-1-train-idx.json')
val_indexes = pd.read_json('../data/module-5-assignment-1-validation-idx.json')


# In[9]:


train_data = loan_data.iloc[train_indexes[0].tolist()]
val_data = loan_data.iloc[val_indexes[0].tolist()]


# In[10]:


safe_loans_raw = train_data[train_data[target] == +1]
risky_loans_raw = train_data[train_data[target] == -1]
print("Number of safe loans  : %s" % len(safe_loans_raw))
print("Number of risky loans : %s" % len(risky_loans_raw))


# In[11]:


# Since there are fewer risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
print(percentage)


# In[12]:


risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(len(risky_loans_raw), random_state=1)


# In[13]:


risky_loans.columns.values


# In[14]:


safe_loans.columns.values


# In[15]:


loans_df = pd.concat([risky_loans, safe_loans])


# In[16]:


loans_df.safe_loans.value_counts()


# In[17]:


loans_df.columns.values


# In[18]:


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


# In[19]:


def get_one_hot_encoding_for_categorical_variables(df):
    categorical_variables = []
    for feat_name, feat_type in zip(df.columns.values, df.dtypes):
        if feat_type == np.dtype('object'):
            categorical_variables.append(feat_name)
            df, _ = encode_target(df, feat_name)
    return df


# In[20]:


loans_df = get_one_hot_encoding_for_categorical_variables(loans_df)


# In[23]:


loans_df.columns.values


# In[24]:


pd.get_dummies(loans_df, columns=['grade']).head()


# In[25]:


loans_df.columns.values


# In[26]:


y = loans_df[target]


# In[27]:


X = loans_df[features]


# In[28]:


features


# In[67]:


big_model = DecisionTreeClassifier(max_depth=10, random_state=1)
big_model.fit(X, y)


# In[29]:


decision_tree_model = DecisionTreeClassifier(max_depth=6, random_state=1)
decision_tree_model.fit(X, y)


# In[30]:


small_model = DecisionTreeClassifier(max_depth=2, random_state=1)
small_model.fit(X, y)


# In[34]:


val_data = get_one_hot_encoding_for_categorical_variables(val_data)
validation_safe_loans = val_data[val_data[target] == 1]
validation_risky_loans = val_data[val_data[target] == -1]


# In[35]:


validation_safe_loans


# In[47]:


isinstance(sample_validation_data_risky, pd.DataFrame)


# In[36]:


sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]


# In[41]:



print(small_model.predict(sample_validation_data_safe[features]))
print(small_model.predict(sample_validation_data_risky[features]))


# In[42]:


print(decision_tree_model.predict(sample_validation_data_safe[features]))
print(decision_tree_model.predict(sample_validation_data_risky[features]))


# In[49]:


sample_validation_data = pd.concat([sample_validation_data_safe, sample_validation_data_risky])


# In[50]:


decision_tree_model.predict_proba(sample_validation_data[features])


# In[51]:


sample_validation_data[target]


# In[52]:


small_model.predict_proba(sample_validation_data[features])


# In[55]:


def visualize_tree(tree, feature_names, model_name):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open(model_name+".dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")


# In[57]:


visualize_tree(small_model, features, "small_model")


# In[58]:


visualize_tree(decision_tree_model, features, "decision_tree_model")


# In[61]:


sample_validation_data[['grade', 'sub_grade',target]]


# In[69]:


small_model.score(val_data[features], val_data[target])


# In[70]:


decision_tree_model.score(val_data[features], val_data[target])


# In[71]:


big_model.score(val_data[features], val_data[target])


# In[77]:


small_model_predictions = small_model.predict(val_data[features])
big_model_predictions = big_model.predict(val_data[features])
decision_tree_model_predictions = decision_tree_model.predict(val_data[features])


# In[78]:


true_labels = val_data[target].as_matrix()


# In[80]:


small_fp = sum((small_model_predictions == 1) & (true_labels == -1))
big_fp = sum((big_model_predictions == 1) & (true_labels == -1))
dt_fp = sum((decision_tree_model_predictions == 1) & (true_labels == -1))


# In[81]:


small_fn = sum((small_model_predictions == -1) & (true_labels == 1))
big_fn = sum((big_model_predictions == -1) & (true_labels == 1))
dt_fn = sum((decision_tree_model_predictions == -1) & (true_labels == 1))


# In[82]:


small_tp = sum((small_model_predictions == 1) & (true_labels == 1))
big_tp = sum((big_model_predictions == 1) & (true_labels == 1))
dt_tp = sum((decision_tree_model_predictions == 1) & (true_labels == 1))


# In[83]:


small_tn = sum((small_model_predictions == -1) & (true_labels == -1))
big_tn = sum((big_model_predictions == -1) & (true_labels == -1))
dt_tn = sum((decision_tree_model_predictions == -1) & (true_labels == -1))


# In[84]:


def get_recall(tp, fn):
    return float(tp)/(tp + fn)


# In[85]:


def get_precision(tp, fp):
    return float(tp)/(tp + fp)


# In[92]:


def get_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


# In[95]:


small_recall = get_recall(small_tp, small_fn)
small_precision = get_precision(small_tp, small_fp)
small_f1 = get_precision(small_precision, small_recall)


# In[91]:




