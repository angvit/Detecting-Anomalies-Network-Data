import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


def load_data(filepath):
    return pd.read_csv(filepath)

def handle_missing_values(df):
    df['attack_cat'] = df['attack_cat'].fillna('Normal').apply(lambda x: x.strip().lower())
    df['ct_flw_http_mthd'].fillna(0, inplace=True)
    df['is_ftp_login'].fillna(0, inplace=True)
    return df

def drop_unnecessary_columns(df):
    return df.drop(columns=['id', 'srcip', 'dstip', 'timestamp'])

def one_hot_encoding(df):
    return pd.get_dummies(df, columns=['protocol', 'service'])

def create_targets(df):
    df['is_anomaly'] = df['label'].apply(lambda x: 1 if x == 1 else 0)
    return df

def split_data(df):
    X = df.drop(columns=['label', 'is_anomaly'])
    y = df['is_anomaly']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def main():
    df = load_data('./datasets/UNSW_NB15_merged.csv')
    df = handle_missing_values(df)
    df = drop_unnecessary_columns(df)
    df = one_hot_encoding(df)
    df = create_targets(df)

    print(df.head())
    print(df.describe())
    print(df.isnull().sum())
    print(df['protocol'].value_counts())
    print(df['service'].value_counts())
    print(df['is_anomaly'].value_counts(normalize=True))
    print(df['attack_cat'].value_counts())

    X_train, X_test, y_train, y_test = split_data(df)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = main()


