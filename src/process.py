import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data(fp):
    return pd.read_csv(fp)

def format_values(df):
    
    df['is_ftp_login'] = df['is_ftp_login'].apply(lambda x: 1 if x >= 1 else 0)
    df['ct_ftp_cmd'] = df['ct_ftp_cmd'].apply(lambda x: 0 if x == ' ' else int(x))

    # preventing error of calling strip() on a float datatype
    # df['attack_cat'] = df['attack_cat'].replace('backdoors', 'backdoor', regex=True)
    # df['attack_cat'] = df['attack_cat'].apply(lambda x: x.strip().lower() if isinstance(x, str) else x)

    df['attack_cat'] = df['attack_cat'].astype(str).str.lstrip('<').str.rstrip('+')
    df['attack_cat'] = df['attack_cat'].replace('backdoors','backdoor', regex=True).apply(lambda x: x.strip().lower())
    df['service'] = df['service'].astype(object)
    df['state'] = df['state'].astype(object)
    return df


def handle_missing_values(df):
    df['attack_cat'] = df['attack_cat'].fillna('Normal').apply(lambda x: x.strip().lower())
    df['ct_flw_http_mthd'].fillna(0, inplace=True)
    df['is_ftp_login'].fillna(0, inplace=True)
    return df

def drop_unnecessary_columns(df):
    # Dropping sport and dsport because of an object/hexadecimal outputting issue
    return df.drop(columns=['srcip', 'dstip', 'sport', 'dsport'])
    # Dropping sport and dsport because of an object/hexadecimal outputting issue
    return df.drop(columns=['srcip', 'dstip', 'sport', 'dsport'])


def create_targets(df):
    df['is_anomaly'] = df['Label'].apply(lambda x: 1 if x == 1 else 0)
    df['is_anomaly'] = df['Label'].apply(lambda x: 1 if x == 1 else 0)
    return df


def correlation_matrix(df):
    correlation_matrix = df.corr(numeric_only=True)
    print(correlation_matrix)

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    plt.title("Feature Correlation Matrix")
    plt.show()


def save_cleaned_csv(df):
    df.to_csv('./datasets/UNSW_NB15_cleaned.csv', index=False)


def main():
    df = load_data('./datasets/UNSW_NB15_merged.csv')
    df = format_values(df)
    df = handle_missing_values(df)
    df = drop_unnecessary_columns(df)
    df = create_targets(df)
    correlation_matrix(df)
    save_cleaned_csv(df)


if '__name__' == '__main__':
    main()


