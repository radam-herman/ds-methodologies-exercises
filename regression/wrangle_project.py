# wrangle_project
# wrangle script for project
# to access zillow DB


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
# %matplotlib inline # only for JNB, not for native python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import env

from sklearn.model_selection import train_test_split
from scipy import stats
from math import sqrt
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

# get URL Fx
def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def wrangle_zillow_bl():

    print("\n "* 4)
    print("BEGINNING OF BASELINE WRANGLE OUTPUTS")

    url = get_db_url('zillow')

    # define SQL Query
    # initially included pred17.transactiondate to manually examine/verify date ranges
    query = '''
    SELECT p2017.calculatedfinishedsquarefeet, p2017.bedroomcnt, p2017.bathroomcnt, p2017.taxvaluedollarcnt
    FROM properties_2017 AS p2017
    JOIN predictions_2017 as pred17 ON p2017.parcelid = pred17.parcelid
    WHERE propertylandusetypeid IN (261, 262, 273, 275, 279) AND pred17.transactiondate > '2017-04-30' AND pred17.transactiondate < '2017-07-01'
    '''

    zillow_project = pd.read_sql(query, url)


    # telco_churn.total_charges.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # call this Fx with:
    # df = wrangle_project.wrangle_zillow()

    zdf = zillow_project
    # X = df_X = df.drop(columns=['customer_id', 'total_charges'])
    # y = df_y = df.total_charges
    return zdf

def bl_sort_X_y(zdf):
    bl_X = zdf[['calculatedfinishedsquarefeet','bedroomcnt', 'bathroomcnt']]
    bl_y = zdf[['taxvaluedollarcnt']]

    return bl_X, bl_y


'''
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.33, random_state=42)
'''
def split_data(X, y, train_pct=.75, random_seed=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_pct, random_state = random_seed)
    return X_train, X_test, y_train, y_test

# standard_scale
def standard_scaler(Xtrain, Xtest):
    Xscaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(Xtrain)
    Xtrain_scaled = pd.DataFrame(Xscaler.transform(Xtrain), columns=Xtrain.columns.values).set_index([Xtrain.index.values])
    Xtest_scaled = pd.DataFrame(Xscaler.transform(Xtest), columns=Xtest.columns.values).set_index([Xtest.index.values])
    return Xscaler, Xtrain_scaled, Xtest_scaled


