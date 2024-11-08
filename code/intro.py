import pandas as pd

def create_features_df():
    features = pd.read_csv('../datasets/UNSW-NB15_features.csv', encoding='latin1')
    return features

features = create_features_df()

def create_merged_df():
    column_names = features['Name']
    df = pd.DataFrame(columns=column_names)
    for i in range(1, 5):
        path = f"../datasets/UNSW-NB15_{str(i)}.csv"
        temp_df = pd.read_csv(path, header=None, low_memory=False)
        temp_df.columns = df.columns
        df = pd.concat([df, temp_df], ignore_index=True)
    return df 

merged_df = create_merged_df()
merged_df.to_csv('../datasets/UNSW_NB15_merged.csv')



