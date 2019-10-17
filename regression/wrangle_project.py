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
    query = '''
    SELECT calculatedfinishedsquarefeet, bedroomcnt, bathroomcnt, taxvaluedollarcnt
    FROM properties_2017
    WHERE propertylandusetypeid IN (261, 262, 275, 279)
    '''

    zillow_project = pd.read_sql(query, url)


    # telco_churn.total_charges.replace(r'^\s*$', np.nan, regex=True, inplace=True)



    # call this Fx with:
    # df = wrangle_project.wrangle_zillow()

    zdf = zillow_project
    # X = df_X = df.drop(columns=['customer_id', 'total_charges'])
    # y = df_y = df.total_charges
    return zdf


