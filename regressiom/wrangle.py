# wrangle.py  
# file for all REGRESSION session

# NOTE TO SELF
    # RUN USING THIS LINE FROM COMMAND LINE
    # python ~/codeup-data-science/ds-methodologies-exercises/regressiom/wrangle.py


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

def get_db_url(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

url = get_db_url('telco_churn')

query = '''
SELECT customer_id, monthly_charges, tenure, total_charges
FROM customers
WHERE contract_type_id = 3;
'''

telco_churn = pd.read_sql(query, url)

telco_churn.head()

print(telco_churn.dtypes)

telco_churn.total_charges.replace(r'^\s*$', np.nan, regex=True, inplace=True)

print(telco_churn.info())

telco_churn['total_charges'] = pd.to_numeric(telco_churn['total_charges'], errors='coerce')

telco_churn.info()

 # change of variable - incase a step goes wrong - original variable is still intact
print("+" * 20)
print("NOTE: variable change to 'telco_churn2")
print("+" * 20)
telco_churn2 = telco_churn.dropna()

telco_churn2.info()



#### AQUIRE and PREP exercises ####

# 1 Acquire customer_id, monthly_charges, tenure, and total_charges
#  from telco_churn database for all customers with a 2 year contract.

# 2 Walk through the steps above using your new dataframe. You may 
# handle the missing values however you feel is appropriate.

# 3 End with a python file wrangle.py that contains the function, 
# wrangle_telco(), that will acquire the data and return 
# a dataframe cleaned with no missing values.