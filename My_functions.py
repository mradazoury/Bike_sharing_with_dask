import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import dask.dataframe as dd
import matplotlib.pyplot as plt
from astral import Astral
import plotly.tools as tls
import plotly
import plotly.plotly as py
from dask_ml.preprocessing import *
import datetime
from dask_ml.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from dask.distributed import Client
from gplearn.genetic import SymbolicTransformer
random_seed = 6666

## Replacing number in season by real names and in weathersit by description
def num_name(df):
    df = df.copy()
    season = {2:'spring', 3:'summer', 4:'fall', 1:'winter'}
    df['season']= df.season.apply(
               lambda x: season[x]).astype('category') 
    weathersit = {1:'Good', 2:'Acceptable', 3:'Bad', 4:'Chaos'}
    df['weathersit']= df.weathersit.apply(
               lambda x: weathersit[x]).astype('category') 
    return df


 ### Dummifying categorical variables
def onehot_encode(df,category):
        df = df.copy()
        numericals = df[list(set(df.columns) - set(category))]
        new_df = numericals.copy()
        categ = df[category].astype('category').categorize()
        enc =DummyEncoder()
        enc = enc.fit_transform(categ)
        categ = enc.repartition(npartitions=8)
        new_df = new_df.repartition(npartitions=8)
        new_df =  dd.concat([new_df,categ], axis=1)
        return new_df

# Preperation for isDaylight()
city_name = 'Washington DC'
a = Astral()
a.solar_depression = 'civil'
city = a[city_name]

def isDaylight(row):
    sun = city.sun(date=row['dteday'] , local=True)
    row['isDaylight'] = 1 if (row['hr'] < sun['sunset'].hour and row['hr'] > sun['sunrise'].hour) else 0
    row['isNoon'] = 1 if row['hr'] == sun['noon'].hour else 0
    return row 

## Creating a new variable that compares the value to the past 7 days 
## the first 5 rows will be dropped if 'windspeed'is calculated and only 2 for the rest 
def relative_values(dataset, columns):
    dataset = dataset.copy()
    max = {'temp':41,'atemp':50,'hum':100,'windspeed':67}
    for i in columns:
        true=dataset[i]*max[i]
        avg7 = true.rolling(min_periods=1,window=24*7).mean().shift()
        std7 = true.rolling(min_periods=1,window=24*7).std().shift()
        name = 'relative_' + i 
        dataset[name]= (true - avg7)/std7
    dataset = dataset.mask(dataset == np.inf, np.nan).dropna()
    dataset = dataset.mask(dataset == -np.inf, np.nan).dropna()
    return dataset 

def addRushHourFlags(row):
    #weekend
    if row['workingday'] == 0 :
        print
        if row['hr'] in [10, 11, 12, 13, 14, 15, 16, 17, 18]:
            row['RushHour-High'] = 1
        elif row['hr'] in [8, 9, 19, 20, 21, 22, 23 ,0]:
            row['RushHour-Med'] = 1
        else:
            row['RushHour-Low'] = 1
    #weekdays
    if row['workingday'] == 1:
        if row['hr'] in [7, 8,9, 16, 17, 18, 19, 20]:
            row['RushHour-High'] = 1
        elif row['hr'] in [6,  10, 11, 12, 13, 15 ,21 ,22 ,23]:
            row['RushHour-Med'] = 1
        else:
            row['RushHour-Low'] = 1
    return row

# def dask_it(df , function):
#         df.apply(lambda x: function(x), axis=1)
#         return df

### This function will calculate the mean of the cnt of the previous 3 weeks during the same hour
def mean_per_hour_3weeks(dataset):
    a = [] 
    for i in range(0,len(dataset)):
        a.append(dataset[ (dataset['instant']>= ((i+1) + -21))
         & ( dataset['instant'] < ( i+1)) &( dataset['hr']  == dataset['hr'][dataset['instant'] == i+1])]['cnt'].mean())
    dataset['mean_per_hour']= dd.from_array(a)
    dataset= dataset.dropna()
    return dataset


def mean_per_hour_3weeks_row(row , dataset):
        i = row['instant']
        row['mean_per_hour'] = dataset[ (dataset.instant >= ((i+1) + -3*7*24))& ( dataset.instant < ( i+1)) &( dataset.hr  == row['hr'])]['cnt'].mean().compute()
        return row

## Genetic programming function that will create new features
def Genetic_P(dataset,target):
        append = 'mean_per_hour'
        a = dataset[append]
        y = dataset[target]
        X=dataset.copy()
        X=X.drop(target,axis=1)
        X=X.drop(append,axis =1)
        with joblib.parallel_backend('dask'):
                function_set = ['add', 'sub', 'mul', 'div',
                        'sqrt', 'log', 'abs', 'neg', 'inv',
                        'max', 'min','sin',
                        'cos',
                        'tan']
                gp = SymbolicTransformer(generations=20, population_size=2000,
                                hall_of_fame=100, n_components=15,
                                function_set=function_set,
                                parsimony_coefficient=0.0005,
                                max_samples=0.9, verbose=1,
                                random_state=random_seed, n_jobs=3)
                gp_features = gp.fit_transform(X,y)
        print('Number of features created out of genetic programing: {}'.format(gp_features.shape))
        gp_dask = dd.from_array(gp_features  )
        new_X = dd.merge(dataset ,gp_dask)
        # n = dd.from_pandas(pd.DataFrame(gp_features) , npartitions=8)
        # # n =n.set_index(dataset.index.values)
        # new_X = pd.concat([dataset, n],axis=1)
        new_X = new_X.dropna()
        return new_X