# prepare.py created out of DATA PREP excercises

# # Prepare Data

# - drop columns
# - fillna
# - split (aka train_test_split into train/test)
# - impute mean, mode, median: SimpleImputer
# - integer encoding: LabelEncoder
# - one hot encoding: OneHotEncoder
# - scale

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def split_data(df): # this function takes in a dataframe - spits out test and train
    df = df.fillna(np.nan)
    seed = 123
    train_prct = .7
    train, test = train_test_split(df, train_size=train_prct, random_state=seed)

    return train, test

