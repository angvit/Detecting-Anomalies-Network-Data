import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_data(filepath):
    return pd.read_csv(filepath)

def handle_missing_values(df):
    df['attack_cat'] = df['attack_cat'].fillna('Normal').apply(lambda x: x.strip().lower())
    df['ct_flw_http_mthd'].fillna(0, inplace=True)
    df['is_ftp_login'].fillna(0, inplace=True)
    return df

def drop_unnecessary_columns(df):
    return df.drop(columns=['srcip', 'dstip'])

def one_hot_encoding(df):
    return pd.get_dummies(df, columns=['proto', 'service'])

def ordinal_encoding(df):
    pass

def create_targets(df):
    df['is_anomaly'] = df['Label'].apply(lambda x: 1 if x == 1 else 0)
    return df

def split_data(df):
    X = df.drop(columns=['Label', 'is_anomaly'])
    y = df['is_anomaly']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test

def random_forest(X_train, X_test, y_train, y_test):
     
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    independent_variables = X_train.columns

    feature_importance_dict = {
        'Feature':independent_variables,
        'Importance': model.feature_importances_
           }
    
    feature_imp = pd.DataFrame.from_dict( feature_importance_dict ).sort_values('feature_importance', ascending=False)
    print(feature_imp)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"{accuracy:.2f}")
    print(classification_report(y_test, y_pred))

def save_cleaned_csv(df):
    pass

def main():
    df = load_data('./datasets/UNSW_NB15_merged.csv')
    df = handle_missing_values(df)
    df = drop_unnecessary_columns(df)
    # df = one_hot_encoding(df)

    df = create_targets(df)

    print(df.head())
    print(df.describe())
    print(df.isnull().sum())

    # print(df['protocol'].value_counts())
    # print(df['service'].value_counts())
    # print(df['is_anomaly'].value_counts(normalize=True))
    # print(df['attack_cat'].value_counts())

    # df.hist(figsize=(15, 15))
    # plt.tight_layout()  
    # plt.show()

    X_train, X_test, y_train, y_test = split_data(df)
    random_forest(X_train, X_test, y_train, y_test)
    

#  X_train, X_test, y_train, y_test = main()

main()


