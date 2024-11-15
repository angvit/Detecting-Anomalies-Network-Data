import numpy as np
import pandas as pd

df = pd.read_csv('./datasets/UNSW_NB15_merged.csv')

print(df.isnull().sum())
