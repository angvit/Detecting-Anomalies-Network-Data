import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import warnings 

warnings.filterwarnings('ignore')

def load_data(fp):
    return pd.read_csv(fp)

def create_targets(df, normal_class_encoded):
    df['is_anomaly'] = df['attack_cat_encoded'].apply(lambda x: 0 if x == normal_class_encoded else 1)
    return df

def label_encoding(df):
    le = LabelEncoder()
    mappings = {}
    for col in ['proto', 'service', 'state', 'attack_cat']:
        df[col + '_encoded'] = le.fit_transform(df[col])
        mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    pd.DataFrame.from_dict(mappings['attack_cat'], orient='index', columns=['Encoded Value']).to_csv('./datasets/attack_cat_mapping.csv')
    df = df.drop(columns=['proto', 'service', 'state', 'attack_cat'])
    return df, mappings['attack_cat']


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
    
    undersampler = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_unsampled = undersampler.fit_resample(X_train, y_train)
    
    return X_train_resampled, X_test, y_train_unsampled, y_test

def grid_search(model, X_train, y_train):
    params = { 
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [20, 22, 24], 
    'min_samples_split': [2, 4, 6]
    }

    grid_search_cv = GridSearchCV(model, param_grid=params, cv=3, n_jobs=-1)
    grid_search_cv.fit(X=X_train, y=y_train)
    return grid_search_cv

def evaluate_model(y_test, y_pred, normal_class_encoded, attack_cat_mapping):
    y_test_binary = np.where(y_test == normal_class_encoded, 0, 1)
    y_pred_binary = np.where(y_pred == normal_class_encoded, 0, 1)

    target_names = attack_cat_mapping.keys()

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Multi-Class Classification Accuracy: {accuracy:0.2f}%")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    cm = confusion_matrix(y_true=y_test_binary, y_pred=y_pred_binary)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
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

    return y_test, y_pred


def main():
    df = load_data('./datasets/UNSW_NB15_cleaned.csv')
    df, attack_cat_mapping = label_encoding(df)
    normal_class_encoded = attack_cat_mapping['normal']
    # print(f"Encoded value for 'normal': {normal_class_encoded}")
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    y_test, y_pred = random_forest(X_train_scaled, X_test_scaled, y_train, y_test)
    evaluate_model(y_test, y_pred, normal_class_encoded, attack_cat_mapping)

# if __name__ == '__main__':
#     main()

main()
