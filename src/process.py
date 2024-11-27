import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
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

def one_hot_encoding(df):
    return pd.get_dummies(df, columns=['proto', 'service', 'state'])

def label_encoding(df):
    le = LabelEncoder()
    for col in ['proto', 'service', 'state', 'attack_cat']:
        df[col + '_encoded'] = le.fit_transform(df[col])
    df = df.drop(columns=['proto', 'service', 'state', 'attack_cat'])
    return df

def standardize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Converting scaled arrays back to Pandas DataFrame and preserving column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled

def create_targets(df):
    df['is_anomaly'] = df['Label'].apply(lambda x: 1 if x == 1 else 0)
    return df

def split_data(df):
    X = df.drop(columns=['Label', 'is_anomaly', 'attack_cat_encoded'])
    y = df['attack_cat_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    return X_train, X_test, y_train, y_test

def random_forest(X_train, X_test, y_train, y_test):

    print("Loading model...")
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    independent_variables = X_train.columns

    feature_importance_dict = {
        'Feature':independent_variables,
        'Importance': model.feature_importances_
    }
    
    feature_imp = pd.DataFrame.from_dict(feature_importance_dict).sort_values('Importance', ascending=False)
    print(feature_imp)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"{accuracy:.2f}")
    print(classification_report(y_test, y_pred))

def save_cleaned_csv(df):
    df.to_csv('./datasets/UNSW_NB15_cleaned.csv', index=False)

def main():
    df = load_data('./datasets/UNSW_NB15_merged.csv')
    df = format_values(df)
    df = handle_missing_values(df)
    df = drop_unnecessary_columns(df)
    df = label_encoding(df)
    df = create_targets(df)

    
    # print(df.isnull().sum())

    # print(df['ct_ftp_cmd'].head())
    # print(df.dtypes)
    # save_cleaned_csv(df)

    # print(df['is_ftp_login'].unique)
    # print(df['is_ftp_login'].value_counts())

    # print(df.state.dtypes)

    # print(df.head())
    # print(df.describe())
    

    # print(df['protocol'].value_counts())
    # print(df['service'].value_counts())
    # print(df['is_anomaly'].value_counts(normalize=True))
    # print(df['attack_cat'].value_counts())

    # df.hist(figsize=(15, 15))
    # plt.tight_layout()  
    # plt.show()

    # correlation_matrix = df.corr(numeric_only=True)
    # print(correlation_matrix)

    # plt.figure(figsize=(12, 10))
    # sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    # plt.title("Feature Correlation Matrix")
    # plt.show()

    X_train, X_test, y_train, y_test = split_data(df)

    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    random_forest(X_train_scaled, X_test_scaled, y_train, y_test)
    

#  X_train, X_test, y_train, y_test = main()

main()


