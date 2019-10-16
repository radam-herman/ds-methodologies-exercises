# REGRESSION - split_scale.py exercise

# Each scaler function should create the object, 
# fit and transform both train and test. 

# They should return the scaler, train df scaled, 
# test df scaled. 
# Be sure your indices represent the original indices
#  from train/test, as those represent 
# the indices from the original dataframe. 
# Be sure to set a random state[sic] where applicable for reproducibility!


# NOTE about Telco data breakout
#       X             |      y
# mo, tenure, cust_id |  total_charge
#
#  X_train, X_test, y_train, y_test


'''
# 1 - split_my_data(X, y, train_pct)
'''

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import math

import wrangle
import env

from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split


def split_my_data(df, train_pct=.75):

    # turn into train, test aka X,y
    seed = 42
    #   df[['name', 'math']]
    X=df[['monthly_charges','tenure']]
    y=df[['total_charges']]
    #X, y = train_test_split(df, train_size = train_pct, random_state = seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=seed)

    return X_train, X_test, y_train, y_test


'''
# 2 standard_scaler()
'''
# Scale to Standard Normal Distribution (mean=0, stdev=1)

    
def standard_scaler(X_train, X_test, y_train, y_test):

    from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
    import math

    # # CREATING SCALERS -
    # only one IFF - only using a train/test setup
        # create SCALER X and SCALER y IFF using deeper (test/train = Xtt/ytt)
    scaler_X = StandardScaler(copy=True, with_mean=True, with_std=True)\
            .fit(X_train)

    scaler_y = StandardScaler(copy=True, with_mean=True, with_std=True)\
            .fit(y_train)

#   # STAGE - Transform Data
    scaled_X_train = pd.DataFrame(scaler_X.transform(X_train),\
        columns=X_train.columns.values).set_index([X_train.index.values])

    scaled_y_train = pd.DataFrame(scaler_y.transform(y_train),\
        columns=y_train.columns.values).set_index([y_train.index.values])

    scaled_X_test = pd.DataFrame(scaler_X.transform(X_test),\
        columns=X_test.columns.values).set_index([X_test.index.values])

    scaled_y_test = pd.DataFrame(scaler_y.transform(y_test),\
        columns=y_test.columns.values).set_index([y_test.index.values])

    return scaler_X, scaler_y, scaled_X_train, scaled_y_train, scaled_X_test, scaled_y_test

'''
# # 3 scale_inverse()
'''
#         # return the scaled data to it's original values using 
#         # scaler.inverse_transform

def scale_inverse(scaler_X, scaler_y, scaled_X_train, scaled_y_train, scaled_X_test, scaled_y_test):
    unscaled_X_train = pd.DataFrame(scaler.inverse_transform(X_train),\
    columns=X_train.columns.values).set_index([X_train.index.values])

    unscaled_y_train = pd.DataFrame(scaler.inverse_transform(y_train),\
    columns=y_train.columns.values).set_index([y_train.index.values])

    unscaled_X_test = pd.DataFrame(scaler.inverse_transform(X_test),\
    columns=X_test.columns.values).set_index([X_test.index.values])

    unscaled_y_test = pd.DataFrame(scaler.inverse_transform(y_test),\
    columns=y_test.columns.values).set_index([y_test.index.values])

    return scaler, unscaled_X_train, unscaled_y_train, unscaled_X_test, unscaled_y_test

# # 4 uniform_scaler()
#  # Scale to Uniform Distribution using the a QuantileTransformer
#     # a non-linear transformer 
#     # It smooths out unusual distributions, and it spreads out 
#     # the most frequent values and reduces the impact of (marginal) outliers

# split_scale.scale_inverse(split_scale.split_my_data(X,y, ))

# # 5 gaussian_scaler()
#  # This uses either the Box-Cox or Yeo-Johnson method 
#  # to transform to resemble normal or standard normal distrubtion
#         # Yeo-Johnson supports both positive or negative data
#         # Box-Cox only supports positive data



# # 6 min_max_scaler()  ## This is a linear transformation  ##
    


# # 7 iqr_robust_scaler()  # Scale data with outliers 
#     # NOTE - With a lot of outliers, scaling using the mean and 
#     #        variance is not going to work very well

df = wrangle.wrangle_telco()
X = df_X = df.drop(columns=['customer_id', 'total_charges'])
y = df_y = df.total_charges
X_train, X_test, y_train, y_test = split_my_data(X, y)
#result = standard_scaler(X_train, X_test, y_train, y_test)
#print(result)

