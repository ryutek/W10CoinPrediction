
# coding: utf-8

# # Windows 10 Coin
# 
# train:  (row: 1,347,190, columns: 1,085)
# test:   (row:   374,136, columns: 1,084)
# 
# y value: if HasClicked == True, app 1.8%
# 
# How to run
# 1. Put the train and test files in ..\input
# 2. Put the script file in ..\script
# 3. In Jupyter Notebook, run all and get submission file in the same script folder

# In[1]:

# Timer and file info
import math
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc # We're gonna be clearing memory a lot
import matplotlib.pyplot as plt
import seaborn as sns
import random
import lightgbm as lgb
import hashlib
#from ml_metrics import mapk
from datetime import datetime
import re
import csv
#import pickle
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from sklearn import model_selection
from sklearn.metrics import matthews_corrcoef, f1_score, classification_report, confusion_matrix, precision_score, recall_score

# Timer
class Timer:
    def __init__(self, text=None):
        self.text = text
        
    def __enter__(self):
        self.cpu = time.clock()
        self.time = time.time()
        if self.text:
            print("{}...".format(self.text))
            print(datetime.now())
        return self

    def __exit__(self, *args):
        self.cpu = time.clock() - self.cpu
        self.time = time.time() - self.time
        if self.text:
            print("%s: cpu %0.2f, time %0.2f\n" % (self.text, self.cpu, self.time))

# Split to train and holdout sets with counts
def sample_train_holdout(_df, sample_count, holdout_count):   
    random.seed(7)
    sample_RowNumber = random.sample(list(_df['RowNumber']), (sample_count + holdout_count))
    train_RowNumber = random.sample(sample_RowNumber, sample_count)
    holdout_RowNumber = list(set(sample_RowNumber) - set(train_RowNumber))
    holdout = _df[_df['RowNumber'].isin(holdout_RowNumber)].copy()
    _df = _df[_df['RowNumber'].isin(train_RowNumber)]
    return _df, holdout 

# Sampling for train and holdout with imbalanced binary label
def trainHoldoutSampling(_df, _id, _label, _seed=7, t_tr=0.5, t_ho=0.5, f_tr=0.05, f_ho=0.5):
    random.seed(_seed)
    
    positive_id = list(_df[_df[_label]==True][_id].values)
    negative_id = list(_df[_df[_label]==False][_id].values)
    
    train_positive_id = random.sample(positive_id, int(len(positive_id) * t_tr))
    holdout_positive_id = random.sample(list(set(positive_id)-set(train_positive_id)), int(len(positive_id) * t_ho)) 
    train_negative_id = random.sample(negative_id, int(len(negative_id) * f_tr))
    holdout_negative_id = random.sample(list(set(negative_id)-set(train_negative_id)), int(len(negative_id) * f_ho))
    
    train_id = list(set(train_positive_id)|set(train_negative_id))
    holdout_id = list(set(holdout_positive_id)|set(holdout_negative_id))
    
    print('train count: {}, train positive count: {}'.format(len(train_id),len(train_positive_id)))
    print('holdout count: {}, holdout positive count: {}'.format(len(holdout_id),len(holdout_positive_id)))
    
    return _df[_df[_id].isin(train_id)], _df[_df[_id].isin(holdout_id)]

def datetime_features2(_df, _col):
    _format='%m/%d/%Y %I:%M:%S %p'
    _df[_col] = _df[_col].apply(lambda x: datetime.strptime(x, _format))
    
    colYear = _col+'Year'
    colMonth = _col+'Month'
    colDay = _col+'Day'
    colHour = _col+'Hour'
    #colYearMonthDay = _col+'YearMonthDay'
    #colYearMonthDayHour = _col+'YearMonthDayHour' 
    
    _df[colYear] = _df[_col].apply(lambda x: x.year)
    _df[colMonth] = _df[_col].apply(lambda x: x.month)
    _df[colDay] = _df[_col].apply(lambda x: x.day)
    _df[colHour] = _df[_col].apply(lambda x: x.hour)
    
    #ymd = [colYear, colMonth, colDay]
    #ymdh = [colYear, colMonth, colDay, colHour]
    
    #_df[colYearMonthDay] = _df[ymd].apply(lambda x: '_'.join(str(x)), axis=1)
    #_df[colYearMonthDayHour] = _df[ymdh].apply(lambda x: '_'.join(str(x)), axis=1)

    return _df
    
# Change date column datetime type and add date time features
def datetime_features(_df, _col, isDelete = False):
    # 1. For years greater than 2017, create year folder with regex and change year to 2017 in datetime column
    # find and return 4 digit number (1st finding) in dataframe string columns
    year_col = _col + 'Year'
    _df[year_col] = _df[_col].apply(lambda x: int(re.findall(r"\D(\d{4})\D", " "+ str(x) +" ")[0]))
    years = sorted(list(_df[year_col].unique()))
    yearsGreaterThan2017 = sorted(i for i in years if i > 2017)

    # Two ways for strange year data (1) change it to 2017 temporarily (2) remove from data; we will go with (1)
    # because we cannot remove test rows anyway
    if isDelete:
        _df = _df[~_df[year_col].isin(yearsGreaterThan2017)]
    else:
        for i in yearsGreaterThan2017:
            print("replace ", i, " to 2017 for conversion")
            _df.loc[_df[year_col] == i, _col] = _df[_df[year_col] == i][_col].values[0].replace(str(i), "2017")
    
    # How to remove strange year rows
    # train = train[~train['year'].isin(yearsGreaterThan2017)]

    # 2. Convert string to datetime
    _df[_col] = pd.to_datetime(_df[_col])
    print(_col, "column conversion to datetime type is done")
    
    # 3. Add more date time features
    month_col = _col + 'Month'
    week_col = _col + 'Week'
    weekday_col = _col + 'Weekday'
    day_col = _col + 'Day'
    hour_col = _col + 'Hour'
    #year_month_day_col = _col + 'YearMonthDay'
    #year_month_day_hour_col = _col + 'YearMonthDayHour'
    
    _df[month_col] = pd.DatetimeIndex(_df[_col]).month
    _df[week_col] = pd.DatetimeIndex(_df[_col]).week
    _df[weekday_col] = pd.DatetimeIndex(_df[_col]).weekday
    _df[day_col] = pd.DatetimeIndex(_df[_col]).day
    _df[hour_col] = pd.DatetimeIndex(_df[_col]).hour
    #_df[year_month_day_col] = _df[[year_col, month_col, day_col]].apply(lambda x: ''.join(str(x)), axis=1)
    #_df[year_month_day_hour_col] = _df[[year_col, month_col, day_col, hour_col]].apply(lambda x: ''.join(str(x)), axis=1)
    print("year, month, week, weekday, day, hour features are added")
    
    return _df

# Delete rows with list condition for dataframe
def delRows(_df, _col, _list):
    _df = _df[~_df[_col].isin(_list)]
    return _df

import re

# Create new column using regex pattern for strings for dataframe
def addFeatureRegex(_df, _col, _newCol):
    _df[_newCol] = _df[_col].apply(lambda x: int(re.findall(r"\D(\d{4})\D", " "+ str(x) +" ")[0]))
    return _df

# Convert string to datetime type
def stringToDatetime(_df, _col):
    _df[_col] = _df[_col].astype('datetime64[ns]')
    return _df

# Add features from datetime
def addDatetimeFeatures(_df, _col):
    _df[_col + 'Year'] = pd.DatetimeIndex(_df[_col]).year
    _df[_col + 'Month'] = pd.DatetimeIndex(_df[_col]).month
    _df[_col + 'Week'] = pd.DatetimeIndex(_df[_col]).week
    _df[_col + 'Weekday'] = pd.DatetimeIndex(_df[_col]).weekday
    _df[_col + 'Day'] = pd.DatetimeIndex(_df[_col]).day
    _df[_col + 'Hour'] = pd.DatetimeIndex(_df[_col]).hour
    return _df

# Get categorical column names
def categoricalColumns(_df):
    cat_columns = _df.select_dtypes(['object']).columns
    print("Categorical column count:", len(cat_columns))
    print("First 5 values:", cat_columns[:5])
    return cat_columns

# Get column names starting with
def columnsStartingWith(_df, _str):
    sorted_list = sorted(i for i in list(_df) if i.startswith(_str))
    print("Column count:", len(sorted_list))
    print("First 5 values:", sorted_list[:5])    
    return sorted_list

# Get column names ending with
def columnsEndingWith(_df, _str):
    sorted_list = sorted(i for i in list(_df) if i.endswith(_str))
    print("Column count:", len(sorted_list))
    print("First 5 values:", sorted_list[:5])    
    return sorted_list

# Get constant columns
def constantColumns(_df):
    constant_list = []
    cols = list(_df) # same as _df.columns.values
    for col in cols:
        if len(_df[col].unique()) == 1:
            constant_list.append(col)
    print("Constant column count:", len(constant_list))
    print("First 5 values:", constant_list[:5])  
    return constant_list

# Add null columns
def makeNullColumns(_df, _cols):
    null_df = _df[_cols].isnull()
    null_df.columns = null_df.columns + 'Null'
    _df = pd.concat([_df, null_df], axis=1)
    return _df

# Union
def union(a, b):
    return list(set(a)|set(b))

def unique(a):
    return list(set(a))

# undersampling - sample rate 0.8 for 80% samling using isUndersampled column 
def underSampling(_df, _sample_rate):
    _df['isUnderSampled'] = 1
    _rand_num = 1/(1-_sample_rate)
    underSample = np.random.randint(_rand_num, size=len(_df[_df['HasClicked'] == 0]))
    _df.loc[_df['HasClicked'] == 0, 'isUnderSampled'] = underSample>0
    return _df

# Add column with value count
def valueCountColumn(_df, _col):
    _dict = dict([(i, a) for i, a in zip(_df[_col].value_counts().index, _df[_col].value_counts().values)])
    _df[_col+'ValueCount'] = _df[_col].apply(lambda x: _dict[x])
    return _df

# Add column with bool values to check if keyword is contained or not
def containColumn(_df, _col, _str):
    _df[_col+'Cotains'+_str] = _df[_col].str.contains(_str)
    return _df

# Feature engineering
def feature_engineering(_df):
    print("shape:", _df.shape)
    print("Add datetime features...")
    datetime_columns = ['BubbleShownTime', 'FirstUpdatedDate', 'OSOOBEDateTime']
    for col in datetime_columns:
        print(col)
        if _df[col].isnull().sum() > 0:
            _df[col] = _df[col].fillna('1/1/2017 11:11:11 AM')
        _df = datetime_features2(_df, col)

    print("shape:", _df.shape)

    gc.collect()
    
    # Null count
    print("Missing value count...")
    _df['CntNs'] = _df.isnull().sum(axis=1) 

    cols = ['AppCategoryNMinus1', 'AppCategoryNMinus2', 'AppCategoryNMinus3', 'AppCategoryNMinus4', 'AppCategoryNMinus5',
           'AppCategoryNMinus6', 'AppCategoryNMinus7', 'AppCategoryNMinus8']
    _df['AppCatCntNs'] = _df[cols].isnull().sum(axis=1)

    #_df[cols] = _df[cols].fillna("NA")
    #for col in cols:
    #    print(col)
    #    _df[col+'HighLevel'] = _df[col].apply(lambda x: str(x).split(':')[0])
   
    # Game segment parse with '.'
    # to-do: 2nd and 3rd parsed values to add as features later, some exception handling is needed
    print("Gamer segment parsing...")
    _df['GamerSegment1'] = _df['GamerSegment'].apply(lambda x: str(x).split('.')[0] if str(x).split('.') else 'Unknown')
    
    # Check creativeName contains keyword or not
    keywords = ['SL', 'TS', 'Week7', 'Meet', 'Skype', 'Battery', 'Switch', 'Performance', 'Security', 'Surge']
    for keyword in keywords:
        _df = containColumn(_df, 'creativeName', keyword)
    #_df['week7'] = _df['Week7'].values + _df['Week 7'].values
    #_df.drop(['Week7', 'Week 7'], axis = 1, inplace = True)
    
    # Convert categorical columns to numeric
    print("Convert categorical columns to numeric...")
    cat_columns = _df.select_dtypes(['object']).columns
    for cat_column in cat_columns:
        print(cat_column)
        if cat_column == 'creativeName':
            _df['creativeNameTest'] = _df['creativeName'].values
        #_df[cat_column] = _df[cat_column].apply(lambda x: abs(hash(x)) )
        _df[cat_column]=_df[cat_column].apply(lambda x: int(hashlib.sha1(str(x).encode('utf-8')).hexdigest(), 16) % (10 ** 16))
    gc.collect()
    
    # Replace missing values with -1
    print("Replace missing values with -1")
    _df = _df.fillna(-1)
    
    # Value count
    print("Value count...")
    cols = ['UniqueUserDeviceKey', 'CampaignId']
    for col in cols:
        print(col)
        _df = valueCountColumn(_df, col)
        
    return _df

# Get best threshold value for F1 score
def f1_best_threshold(_actual, _pred):
    thresholds = np.linspace(0.01, 0.5, 1000)

    fc = np.array([f1_score(_actual, _pred>thr) for thr in thresholds])
    plt.plot(thresholds, fc)
    best_threshold = thresholds[fc.argmax()]
    print('f1 score:', fc.max())
    print('best threshold:', best_threshold)
    print('TF pred mean:', (_pred>best_threshold).mean())
    
    return best_threshold


# In[7]:

# Read tsv file
test2 = pd.read_csv('CoinMlCompetitionSoftlandingEvaluateNoLabel.tsv', sep='\t', header = None)
#print(test2.head())

# Add header because test does not header
df_header = pd.read_csv('test_header.csv')
test_header2 = df_header['0'].values
#print('test header', test_header2)

test2.columns = test_header2

# Reduce test size by leaving train features only
df_initial_features = pd.read_csv('initial_features.csv')
initial_features2 = df_initial_features['0'].values
#print('initial features', initial_features2)

test2 = test2[list(set(initial_features2) - set(['HasClicked']))]

# Feature engineering - should not delete odd date rows
# test = feature_engineering(test, isDeleteOddDateRows=False)
test2 = feature_engineering(test2)

random.seed(2007)
bst2 = lgb.Booster(model_file="model.txt") 

# Predict test
df_final_features = pd.read_csv('final_features.csv')
final_features2 = df_final_features['0'].values
#print('final features:', final_features2)

preds = bst2.predict(test2[final_features2], num_iteration=300)

# Best threshold from train
#val_best_threshold = 0.072
val_best_threshold = 0.0634634634635

# Create submissin file
test_id = test2.RowNumber.values
submission = pd.DataFrame({'RowNumber': test_id})
submission['HasClicked'] = preds > val_best_threshold
print("Click mean:", submission.HasClicked.mean())
print("Submission file...")
submission.to_csv("W10_Coin_test_prediction_0403.csv", index = False)
submission.head()

# click mean: 0.0288931594576


# In[ ]:




# In[ ]:



