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



# 1 - split_my_data(X, y, train_pct)

    # Create TRAIN & TEST Objects

import pandas as pd
import numpy as np
import wrangle
from sklearn.model_selection import train_test_split

# df = wrangle.wrangle_telco()

# X = df_X = df.drop(columns=['customer_id', 'total_charges'])
# y = df_y = df.total_charges


    # call this function fm ipython terminal,
        # first provide X and y variables
    # X = df_X = df.drop(columns=['customer_id', 'total_charges'])
    # y = df_y = df.total_charges
        # NOTE: pct_train the last varialbe is not provided, there is a default
        # simply provide alternate train set size if desired
        # example:
    # split_scale.split_my_data(X,y, )

def split_my_data(X, y, pct_train = .75):
#def split_my_data():

    X_train, X_test, y_train, y_test = train_test_split(X, y, pct_train, random_state=42)

    # print("\n "* 1)
    # print(" ======= X_ train and test shapes =======")
    # print(X_train.shape); print(X_test.shape)

    # print("\n "* 1)
    # print(" ======= y_ train and test shapes =======")
    # print(y_train.shape); print(y_test.shape)

    return X_train, X_test, y_train, y_test


#X_train, X_test, y_train, y_test = train_test_split(wrangle, y, train_size = .75, random_state=42)
#train, test = train_test_split(wrangle, train_size = .75, random_state = 123)



# 2 standard_scaler() 
# Scale to Standard Normal Distribution (mean=0, stdev=1)

    # STAGE - Create Object and Fit to Data
    

# from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
# import math

# # CREATING SCALERS -
#     # converting all to dataframes
# #X_train_df = pd.DataFrame(X_train)
# #X_test_df = pd.DataFrame(X_test)
#    # y is just a series and gets interpreted only as a series and not a DF as is needed
# y_train_df = pd.DataFrame(y_train)
# y_test_df = pd.DataFrame(y_test)

# # X_train scaler
# scaler_X_train = StandardScaler(copy=True, with_mean=True, with_std=True)\
#     .fit(X_train[['monthly_charges','tenure']])

# print("X_train")
# print("\n "* 1)
# print("Mean:") 
# print(scaler_X_train.mean_)
# print("Standard Deviation:")
# print([math.sqrt(i) for i in scaler_X_train.var_])

# # X_test scaler
# scaler_X_test = StandardScaler(copy=True, with_mean=True, with_std=True)\
#     .fit(X_test[['monthly_charges','tenure']])

# print("X_test")

# print("\n "* 1)
# print("Mean:") 
# print(scaler_X_test.mean_)
# print("Standard Deviation:")
# print([math.sqrt(i) for i in scaler_X_test.var_])

# # y_train scaler
# scaler_y_train = StandardScaler(copy=True, with_mean=True, with_std=True)\
#     .fit(y_train_df)

# print("y_train")

# print("\n "* 1)
# print("Mean:") 
# print(scaler_y_train.mean_)
# print("Standard Deviation:")
# print([math.sqrt(i) for i in scaler_y_train.var_])

# # y_test scaler
# scaler_y_test = StandardScaler(copy=True, with_mean=True, with_std=True)\
#     .fit(y_test_df)

# print("y_test")

# print("\n "* 1)
# print("Mean:") 
# print(scaler_y_test.mean_)
# print("Standard Deviation:")
# print([math.sqrt(i) for i in scaler_y_test.var_])


#   # STAGE - Transform Data











# # 3 scale_inverse()
#         # return the scaled data to it's original values using 
#         # scaler.inverse_transform


# # 4 uniform_scaler()
#  # Scale to Uniform Distribution using the a QuantileTransformer
#     # a non-linear transformer 
#     # It smooths out unusual distributions, and it spreads out 
#     # the most frequent values and reduces the impact of (marginal) outliers



# # 5 gaussian_scaler()
#  # This uses either the Box-Cox or Yeo-Johnson method 
#  # to transform to resemble normal or standard normal distrubtion
#         # Yeo-Johnson supports both positive or negative data
#         # Box-Cox only supports positive data



# # 6 min_max_scaler()  ## This is a linear transformation  ##
    


# # 7 iqr_robust_scaler()  # Scale data with outliers 
#     # NOTE - With a lot of outliers, scaling using the mean and 
#     #        variance is not going to work very well



