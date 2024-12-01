import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import warnings 

warnings.filterwarnings('ignore')

def load_data(fp):
    return pd.read_csv(fp)


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


def split_data(df):
    X = df.drop(columns=['is_anomaly', 'attack_cat_encoded'])
    y = df['attack_cat_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    return X_train, X_test, y_train, y_test

def grid_search(model, X_train, y_train):
    params = { 
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [20, 22, 24], 
    'min_samples_split': [2, 4, 6]
    }

    grid_search_cv = GridSearchCV(model, param_grid=params, cv=3, n_jobs=-1)
    grid_search_cv.fit(X=X_train, y=y_train)
    return grid_search_cv

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{accuracy:0.2f}%")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=['Non-Anomaly', 'Anomaly'], yticklabels=['Non-Anomaly', 'Anomaly'])
    plt.title('Confusion Matrix for Random Forest Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def random_forest(X_train, X_test, y_train, y_test):
    print("Loading model...")

    # grid_search_cv = grid_search(model, X_train, y_train)
    # model = grid_search_cv.best_estimator_
    #print(f"Best Model: {model}")

    model = RandomForestClassifier(criterion='gini', max_depth=22, min_samples_split=6, n_estimators=300, n_jobs=-1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    independent_variables = X_train.columns

    feature_importance_dict = {
        'Feature':independent_variables,
        'Importance': model.feature_importances_
    }
    
    feature_imp = pd.DataFrame.from_dict(feature_importance_dict).sort_values('Importance', ascending=False)
    print(feature_imp)

    evaluate_model(y_test, y_pred)


def main():
    df = load_data('./datasets/UNSW_NB15_cleaned.csv')
    df = label_encoding(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    random_forest(X_train_scaled, X_test_scaled, y_train, y_test)


if '__name__' == '__main__':
    main()


main()