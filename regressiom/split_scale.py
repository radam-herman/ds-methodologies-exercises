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

    # Create Object and Fit to Data
    # Transform Data

import pandas as pd
import numpy as np
import wrangle
from sklearn.model_selection import train_test_split

df = wrangle.wrangle_telco()

print("\n "* 2)
print("+" * 20)
print("++++++++++ DF ++++++++++++")
print("\n "* 1)
print(df.head())

X = df_X = df.drop(columns=['customer_id', 'total_charges'])

y = df_y = df.total_charges

print("\n "* 1)
print(" ======= X header info =======")
print(X.head())

print("\n "* 1)
print(" ======= y header info =======")
print(y.head())


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .75, random_state=42)

print("\n "* 1)
print(" ======= X_train header info =======")
print(X_train.head())

print("\n "* 1)
print(" ======= X_test header info =======")
print(X_test.head())

print("\n "* 1)
print(" ======= X_test header info =======")
print(y_train.head())

print("\n "* 1)
print(" ======= X_test header info =======")
print(y_test.head())

#X_train, X_test, y_train, y_test = train_test_split(wrangle, y, train_size = .75, random_state=42)
#train, test = train_test_split(wrangle, train_size = .75, random_state = 123)



# 2 standard_scaler() 
# Scale to Standard Normal Distribution (mean=0, stdev=1)

    # Create Object and Fit to Data
    # Transform Data


# 3 scale_inverse()
        # return the scaled data to it's original values using 
        # scaler.inverse_transform


# 4 uniform_scaler()
 # Scale to Uniform Distribution using the a QuantileTransformer
    # a non-linear transformer 
    # It smooths out unusual distributions, and it spreads out 
    # the most frequent values and reduces the impact of (marginal) outliers



# 5 gaussian_scaler()
 # This uses either the Box-Cox or Yeo-Johnson method 
 # to transform to resemble normal or standard normal distrubtion
        # Yeo-Johnson supports both positive or negative data
        # Box-Cox only supports positive data



# 6 min_max_scaler()  ## This is a linear transformation  ##
    


# 7 iqr_robust_scaler()  # Scale data with outliers 
    # NOTE - With a lot of outliers, scaling using the mean and 
    #        variance is not going to work very well



