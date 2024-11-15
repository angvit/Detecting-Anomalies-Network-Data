import numpy as np
import pandas as pd

csv_path = './datasets/UNSW_NB15_merged.csv'

df = pd.read_csv('./datasets/UNSW_NB15_merged.csv')

# Check for null values
print(df.isnull().sum())

# Replace null values in attack_cat with the label 'No Attack'
df['attack_cat'].fillna('No Attack', inplace=True)

print(df.isnull().sum())

### NOTES
# THERE ARE NO DUPLICATES
# print(df[df.duplicated()])

# Standardize data types
# print(df.info())

# Change dtype of dsport column to int
# df['dsport'] = pd.to_numeric(df['dsport'], errors='coerce')
# WEIRD: dsport contains non-integer values

# print(df.info())
# print(df.isnull().sum())
# print(df[df['dsport'].isnull()])