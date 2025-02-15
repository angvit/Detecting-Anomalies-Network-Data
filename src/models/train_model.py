import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.utils import resample
import pickle
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


def reduce_normal_class(df, target_size):
    normal_class = df[df['attack_cat_encoded'] == 6]
    attack_class = df[df['attack_cat_encoded'] != 6]
    
    normal_class_reduced = normal_class.sample(n=target_size)
    balanced_df = pd.concat([normal_class_reduced, attack_class], ignore_index=True)
    
    return balanced_df


def reduce_normal_class_alt(df):
    normal_class = df[df['attack_cat_encoded'] == 6]
    attack_class = df[df['attack_cat_encoded'] != 6]

    # downsampling the majority class to match minority size.
    normal_class_reduced = resample(normal_class, replace=False, n_samples=len(attack_class), random_state=42)
    balanced_df = pd.concat([normal_class_reduced, attack_class])
    
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
    return balanced_df


def save_best_model(model, accuracy_scores):
    best_model_index = np.argmax(accuracy_scores)
    best_model = model[best_model_index]
    
    output_fp = './app/model/rf_model.pkl'

    with open(output_fp, 'wb') as f:
        pickle.dump(best_model, f)
    
    print("Model has been saved.")


def stratified_k_fold_cv(df, attack_cat_mapping_dict):
    X = df.drop(columns=['is_anomaly', 'attack_cat_encoded'])
    y = df['attack_cat_encoded']

    target_names = attack_cat_mapping_dict.keys()
    feature_names = X.columns

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
    
    accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], []  
    feature_importances_lst = []
    fold_results = []
    models = []
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
        model, y_test, y_pred, feature_imp = random_forest(X_train_scaled, X_test_scaled, y_train, y_test)
        
        models.append(model)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        fold_results.append((y_test, y_pred))
        feature_importances_lst.append(feature_imp)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    save_best_model(models, accuracy_scores)

    print(f"Fold Accuracy {accuracy:.2%}")
    print(classification_report(y_test, y_pred, target_names=target_names))

    return accuracy_scores, precision_scores, recall_scores, f1_scores, feature_importances_lst, fold_results, feature_names


def grid_search(model, X_train, y_train):
    params = { 
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [20, 22, 24], 
    'min_samples_split': [2, 4, 6]
    }

    grid_search_cv = GridSearchCV(model, param_grid=params, cv=3, n_jobs=-1)
    grid_search_cv.fit(X=X_train, y=y_train)
    return grid_search_cv


def random_search(model, X_train, y_train):
    params = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
    }

    random_search_cv = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=50, cv=3, n_jobs=-1)
    random_search_cv.fit(X_train, y_train)
    return random_search_cv


def calculate_average_score(eval_metric_name, eval_metric_lst):
    print(f"Average {eval_metric_name} across all folds: {sum(eval_metric_lst) / len(eval_metric_lst):.2%}")
    return


def evaluate_model(accuracy_scores, precision_scores, recall_scores, f1_scores, feature_importances_lst, fold_results, attack_cat_mapping_dict, feature_names):

    best_fold_index = np.argmax(accuracy_scores)
    y_test_best, y_pred_best = fold_results[best_fold_index]

    avg_feature_importances = np.mean(feature_importances_lst, axis=0)

    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': avg_feature_importances
    }).sort_values(by='Importance', ascending=False)

    calculate_average_score("accuracy score", accuracy_scores)
    calculate_average_score("precision score", precision_scores)
    calculate_average_score("recall score", recall_scores)
    calculate_average_score("f1 score", f1_scores)
    
    feature_importances.to_csv('././datasets/feature_importances.csv', index=False)
    print(feature_importances)

    y_test_binary = np.where(y_test_best == attack_cat_mapping_dict['normal'], 0, 1)
    y_pred_binary = np.where(y_pred_best == attack_cat_mapping_dict['normal'], 0, 1)
    
    cm = confusion_matrix(y_true=y_test_binary, y_pred=y_pred_binary)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix for Random Forest Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def random_forest(X_train, X_test, y_train, y_test):
    print("Loading model...")

    model = RandomForestClassifier(criterion='gini', max_depth=22, min_samples_split=6, n_estimators=300, n_jobs=-1, class_weight="balanced")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    feature_imp = model.feature_importances_

    return model, y_test, y_pred, feature_imp


def main():
    df = load_data('././datasets/UNSW_NB15_cleaned.csv')
    attack_cat_mapping_df = load_data('././datasets/attack_cat_mapping.csv')

    # df, attack_cat_mapping = label_encoding(df)
    attack_cat_mapping_dict = attack_cat_mapping_df.set_index('Unnamed: 0')['Encoded Value'].to_dict()
    print("Current attack_cat_mapping_dict:\n", attack_cat_mapping_dict)
    # normal_class_encoded = attack_cat_mapping_dict['normal']
    balanced_df = reduce_normal_class_alt(df)

    accuracy_scores, precision_scores, recall_scores, f1_scores, feature_importances_lst, fold_results, feature_names = stratified_k_fold_cv(balanced_df, attack_cat_mapping_dict)
    evaluate_model(accuracy_scores, precision_scores, recall_scores, f1_scores, feature_importances_lst, fold_results, attack_cat_mapping_dict, feature_names)


if __name__ == '__main__':
    main()


