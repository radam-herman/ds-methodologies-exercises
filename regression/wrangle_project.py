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

from scipy import stats
from math import sqrt

# get URL Fx
def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def wrangle_zillow():

    print("\n "* 4)
    print("BEGINNING OF WRANGLE OUTPUTS")

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

def mvp_sort_X_y(zdf):
    mvp_X = zdf[['calculatedfinishedsquarefeet','bedroomcnt', 'bathroomcnt']]
    mvp_y = zdf[['taxvaluedollarcnt']]

    return mvp_X, mvp_y


