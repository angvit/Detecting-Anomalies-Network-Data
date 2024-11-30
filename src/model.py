import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


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
    X = df.drop(columns=['Label', 'is_anomaly', 'attack_cat_encoded'])
    y = df['attack_cat_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    return X_train, X_test, y_train, y_test

def grid_search(model, X_train, y_train):
    params = { 
    'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
    'max_features': ['auto', 'sqrt'], 
    'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)], 
    'min_samples_split': [2, 5, 10],
    'max_leaf_nodes': [5, 10]
    }

    grid_search_cv =  GridSearchCV(model, param_grid= params)
    grid_search_cv.fit(X=X_train, y=y_train)
    print(grid_search_cv.best_estimator_) 


def random_forest(X_train, X_test, y_train, y_test):
    print("Loading model...")
    grid_search_cv =  grid_search(model)
    
    model = grid_search_cv.best_estimator_
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
    print(f"{accuracy:0.2f}%")
    print(classification_report(y_test, y_pred))


def main():
    df = load_data('./datasets/UNSW_NB15_cleaned.csv')
    df = label_encoding(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    random_forest(X_train_scaled, X_test_scaled, y_train, y_test)


if '__name__' == '__main__':
    main()