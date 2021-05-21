# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# LIBRARIES 
import pandas as pd
import numpy as np
import os
from os import path
import xlsxwriter
from pathlib import Path
import csv
import sys
import altair as alt
from datetime import date
from datetime import datetime
#import matplotlib.pyplot as plt
#import pandas_profiling
print("+++ libraries loaded!")


# %%
### ---- 
### DATE AND TIME VARIABLES
### ---- 
today = date.today()
print (today)

from datetime import date
from datetime import datetime

today = date.today()

# ddmmYY_HHMMSS

now = datetime.now()
dt_string = now.strftime("%m%d%Y_%H%M%S")
print("SUFFIX date and time =", dt_string)

# dd/mm/YY
d1 = today.strftime("%d/%m/%Y")
print("d1 =", d1)

# Textual month, day and year	
d2 = today.strftime("%B %d, %Y")
print("d2 =", d2)

# mm/dd/y
d3 = today.strftime("%m/%d/%y")
print("d3 =", d3)

# Month abbreviation, day and year	
d4 = today.strftime("%b-%d-%Y")
print("d4 =", d4)

# datetime object containing current date and time
now = datetime.now()
print("now =", now)

# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

# ddmmYY_HHMMSS
dt_string = now.strftime("%m%d%Y_%H%M%S")
print("date and time =", dt_string)


current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

current_time = now.strftime("%H_%M_%S")
print("Current Time =", current_time)


now = datetime.now().time() # time object
print("now =", now)
print("type(now) =", type(now))	


# %%
## ------
## FILES 
## ------
home_dir = './'
sub_dir
work_dir = os.path.join(home_dir, sub_dir)
csv_files = [os.path.join(work_dir , f) for f in os.listdir(work_dir) if (".csv" in f and "~" not in f) ]
print(csv_files)


# %%
# FILES Alternate 
# files = os.listdir(path)
# for f in files:
# 	print(f)
#   print(os.path.join(root, name))

# directory = os.path.join("c:\\","path")
# for root,dirs,files in os.walk(directory):
#     for file in files:
#        if file.endswith(".csv"):
#            f=open(file, 'r')
#            #  perform calculation
#            f.close()

# print(files)


# %%
## --- 
## CREATE DF form CSV 
## --- 
# CONCATENATE many csv files in one df
df_raw = pd.concat((pd.read_csv(f) for f in csv_files))


# %%
## ---
## DF ANALYSIS 
## ---
df           # print the first 30 and last 30 rows
type(df)     # DataFrame
df.head()    # print the first 5 rows
df.head(10)  # print the first 10 rows
df.tail()    # print the last 5 rows
df.index     # “the index” (aka “the labels”)
df.columns   # column names (which is “an index”)
df.dtypes    # data types of each column
df.shape     # number of rows and columns
# underlying numpy array — df are stored as numpy arrays for effeciencies.
df.values
df['Market Code'].value_counts()
df['Market Code'].unique()


del df_resp # Delete 


# %%
### ----
# DF COLUMN ANALYSIS
### ----

df[‘column_y’]         # select one column
type(df[‘column_y’])   # determine datatype of column (e.g., Series)

# summarize (describe) the DataFrame
df.describe()          # describe all numeric columns
df.describe(include=[‘object’])  # describe all object columns
df.describe(include=’all’)      # describe all columns

#filter df by one column, and print out values of another column
#when using numeric values, no quotations
df[df.column_y == “string_value”].column_z
df[df.column_y == 20].column_z

# display only the number of rows of the ‘df’ DataFrame
df.shape[0]
# display the 3 most frequent occurances of column in ‘df’
df.column_y.value_counts()[0:3]


# %%
### --- 
### DF FILTER
### ---- 
# boolean filtering: only show df with column_z < 20
filter_bool = df.column_z < 20    # create a Series of booleans…
df[filter_bool]                # …and use that Series to filter rows
df[filter_bool].describe()     # describes a data frame filtered by filter_bool
df[df.column_z < 20]           # or, combine into a single step
df[df.column_z < 20].column_x  # select one column from the filtered results
df[df[“column_z”] < 20].column_x     # alternate method
# value_counts of resulting Series, can also use .mean(), etc. instead of .value_counts()
df[df.column_z < 20].column_x.value_counts()


# %%
### --- 
### DROP 
### --- 

df = df.drop(some labels)
df = df.drop(df[<some boolean condition>].index)

## Use dropna with parameter subset for specify column for check NaNs:
# 1. Dropping columns
# The drop function is used to drop columns and rows. We pass the labels of rows or columns to be dropped.
df.drop(['RowNumber', 'CustomerId', 'Surname', 'CreditScore'], axis=1, inplace=True)


# %%
## --- 
## APPLY an IF condition in Pandas DataFrame
## https://datatofish.com/if-condition-in-pandas-dataframe/
## --- 
### 1 
### If the number is equal or lower than 4, then assign the value of ‘True’skus_resp_list
### Otherwise, if the number is greater than 4, then assign the value of ‘False’
df.loc[df['column name'] condition, 'new column name'] = 'value if condition is met'
df.loc[df['set_of_numbers'] <= 4, 'equal_or_lower_than_4?'] = 'True' 
df.loc[df['set_of_numbers'] > 4, 'equal_or_lower_than_4?'] = 'False' 

### 2 (2) IF condition – set of numbers and lambda
df['new column name'] = df['column name'].apply(lambda x: 'value if condition is met' if x condition else 'value if condition is not met')
df['equal_or_lower_than_4?'] = df['set_of_numbers'].apply(lambda x: 'True' if x <= 4 else 'False')

### 3) IF condition – strings
### If the name is equal to ‘Bill,’ then assign the value of ‘Match’
### Otherwise, if the name is not ‘Bill,’ then assign the value of ‘Mismatch’
df.loc[df['First_name'] == 'Bill', 'name_match'] = 'Match'  
df.loc[df['First_name'] != 'Bill', 'name_match'] = 'Mismatch'  

### 4) IF condition – strings and lambada 
df['name_match'] = df['First_name'].apply(lambda x: 'Match' if x == 'Bill' else 'Mismatch')

### 5) IF condition with OR
df.loc[(df['First_name'] == 'Bill') | (df['First_name'] == 'Emma'), 'name_match'] = 'Match'  
df.loc[(df['First_name'] != 'Bill') & (df['First_name'] != 'Emma'), 'name_match'] = 'Mismatch'  


# %%



# %%



# %%
## --- 
## 30 Examples to Master Pandas
## https://towardsdatascience.com/30-examples-to-master-pandas-f8a2da751fa4
## --- 
# create a DataFrame from a dictionary
pd.DataFrame({‘column_x’: [‘value_x1’, ‘value_x2’, ‘value_x3’], ‘column_y’: [‘value_y1’, ‘value_y2’, ‘value_y3’]})
# create a DataFrame from a list of lists
pd.DataFrame([[‘value_x1’, ‘value_y1’], [‘value_x2’, ‘value_y2’], [‘value_x3’, ‘value_y3’]], columns=[‘column_x’, ‘column_y’])

# %% [markdown]
# 30 Examples to Master Pandas
# https://towardsdatascience.com/30-examples-to-master-pandas-f8a2da751fa4

# %%


# %% [markdown]
# How to Create Pandas DataFrame in Python 
# https://datatofish.com/create-pandas-dataframe/
# https://datatofish.com/python-tutorials/ 

# %%
## --- 
## How to Create Pandas DataFrame in Python 
## https://datatofish.com/create-pandas-dataframe/
## --- 
# 1 - typing values in Python to create Pandas DataFrame
data = {'First Column Name':  ['First value', 'Second value',...],
        'Second Column Name': ['First value', 'Second value',...],
         ....}
df = pd.DataFrame (data, columns = ['First Column Name','Second Column Name',...])

# OR  
cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4'],
        'Price': [22000,25000,27000,35000]}
df = pd.DataFrame(cars, columns = ['Brand', 'Price'])

#2 Method 2: importing values from an Excel file to create Pandas DataFrame
data = pd.read_excel(r'Path where the Excel file is stored\File name.xlsx') #for an earlier version of Excel use 'xls'
df = pd.DataFrame(data, columns = ['First Column Name','Second Column Name',...])
# ImportError: Install xlrd >= 1.0.0 for Excel support
pip3 install xlrd


# Get the maximum value from the DataFrame
max1 = df['Price'].max()


# %%
### ADD multiple columns to pandas dataframe in one assignment

# 1) Three assignments in one, using list unpacking:
df['column_new_1'], df['column_new_2'], df['column_new_3'] = [np.nan, 'dogs', 3]

#2) DataFrame conveniently expands a single row to match the index, so you can do this:
df[['column_new_1', 'column_new_2', 'column_new_3']] = pd.DataFrame([[np.nan, 'dogs', 3]], index=df.index)

#3) Make a temporary data frame with new columns, then combine with the original data frame later:
df = pd.concat(
    [
        df,
        pd.DataFrame(
            [[np.nan, 'dogs', 3]], 
            index=df.index, 
            columns=['column_new_1', 'column_new_2', 'column_new_3']
        )
    ], axis=1
)

#4) Similar to the previous, but using join instead of concat (may be less efficient):
df = df.join(pd.DataFrame(
    [[np.nan, 'dogs', 3]], 
    index=df.index, 
    columns=['column_new_1', 'column_new_2', 'column_new_3']
))

#5) Using a dict is a more "natural" way to create the new data frame than the previous two, but the new columns will be sorted alphabetically (at least before Python 3.6 or 3.7):
df = df.join(pd.DataFrame(
    {
        'column_new_1': np.nan,
        'column_new_2': 'dogs',
        'column_new_3': 3
    }, index=df.index
))

#6) Use .assign() with multiple column arguments.
#I like this variant on @zero's answer a lot, but like the previous one, the new columns will always be sorted alphabetically, at least with early versions of Python:
df = df.assign(column_new_1=np.nan, column_new_2='dogs', column_new_3=3)

#7) This is interesting (based on https://stackoverflow.com/a/44951376/3830997), but I don't know when it would be worth the trouble:
new_cols = ['column_new_1', 'column_new_2', 'column_new_3']
new_vals = [np.nan, 'dogs', 3]
df = df.reindex(columns=df.columns.tolist() + new_cols)   # add empty cols
df[new_cols] = new_vals  # multi-column assignment works for existing cols

#8) In the end it's hard to beat three separate assignments:
df['column_new_1'] = np.nan
df['column_new_2'] = 'dogs'
df['column_new_3'] = 3


# %%



# %%
df = pd.DataFrame(columns=['lib', 'qty1', 'qty2'])
for i in range(5):
    df.loc[i] = ['name' + str(i)] + list(randint(10, size=2))

df
     lib qty1 qty2
0  name0    3    3
1  name1    2    4
2  name2    2    8
3  name3    2    1
4  name4    9    6


# %%
def func(row):
   if row['a'] == "3":
        row2 = row.copy()
        # make edits to row2
        return pd.concat([row, row2], axis=1)
   return row

pd.concat([func(row) for _, row in df.iterrows()], ignore_index=True, axis=1).T


# %%
def row_appends(x):
    newrows = x.loc[x['a'].isin(['3', '4', '5'])].copy()
    newrows.loc[x['a'] == '3', 'b'] = 10  # make conditional edit
    newrows.loc[x['a'] == '4', 'b'] = 20  # make conditional edit
    newrows.index = newrows.index + 0.5
    return newrows

res = pd.concat([df, df.pipe(row_appends)])        .sort_index().reset_index(drop=True)


# %%



# %%
# https://stackoverflow.com/questions/10715965/create-pandas-dataframe-by-appending-one-row-at-a-time 
### This is The Right Way™ to accumulate your data

data = []
for a, b, c in some_function_that_yields_data():
    data.append([a, b, c])

df = pd.DataFrame(data, columns=['A', 'B', 'C'])


# %%



# %%
market_store = {'CA':ca_test_store,'US':us_test_store}
#df_bev['store'] = df_bev['Market Code'].map(market_store)


# %%
people = [
{'name': "Tom", 'age': 10},
{'name': "Mark", 'age': 5},
{'name': "Pam", 'age': 7}
]

filter(lambda person: person['name'] == 'Pam', people)
result (returned as a list in Python 2):

[{'age': 7, 'name': 'Pam'}]


# %%



# %%



# %%



# %%



