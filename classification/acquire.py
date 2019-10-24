
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import env

#from sklearn.datasets import load_iris # for iris data
from sklearn import datasets

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_titanic_data():
    return pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))

def get_iris_data():  # THIS KEPT GETTING A "Exception: Data must be 1-dimensional"

    query = '''
SELECT * 
FROM species as s
JOIN measurements as m ON s.species_id = m.species_id
    '''
    return pd.read_sql(query, get_connection('iris_db'))

def get_iris_data_sklearn_edition():
    iris = datasets.load_iris()
    return iris