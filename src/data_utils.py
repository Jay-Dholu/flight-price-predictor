# this file contains functions for loading and splitting data


import pandas as pd


def load_data(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df


def split_data(data):
    x_data = data.drop(columns='Price')
    y_data = data['Price'].copy()

    return x_data, y_data
