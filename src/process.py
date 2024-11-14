import os
import pandas as pd

# This code snipet is to prevent the cvs from being read in everytime you run the file
# Here we are creating a pickle file if it doesnt exsist and if it does then it just loads the df from that
csv_path = './datasets/UNSW_NB15_merged.csv'
cache_path = './datasets/UNSW_NB15.pkl'

def load_data():
    global df
    if os.path.exists(cache_path):
        print("Loading data from cache...")
        df = pd.read_pickle(cache_path)
    else:
        print("Loading data from CSV and creating cache...")
        df = pd.read_csv(csv_path)
        df.to_pickle(cache_path)
    return df

# call the function
df = load_data()

################################# Columns with missing values is_ftp_login,ct_flw_http_mthd ,attack_cat  #####################################################
print('These are all the unique values for is_ftp_login:', df['is_ftp_login'].unique())
print('These are all the unique values for ct_flw_http_mthd:', df['ct_flw_http_mthd'].unique())
# These are all the unique values for is_ftp_login: [ 0.  1. nan  2.  4.]
# These are all the unique values for ct_flw_http_mthd: [ 0.  1.  2.  4. 14.  8.  6. 12. 10.  3.  5. 36.  9. nan 16. 25. 30.]


################################ Fill in all the 'missing' values with "No Attack" ####################################################################
df['attack_cat'].fillna('No Attack', inplace=True)

############################################ drop the columns with the missing values ####################################################### 
#df.drop(columns=['is_ftp_login', 'ct_flw_http_mthd'], inplace=True) # drop these columns in the dataframe and not return a new one


############################################ Find all the columns that are of type object #####################################################################
object_df = df.select_dtypes(include=['object'])
print(object_df.info())

# Data columns (total 9 columns):
#  #   Column      Dtype
# ---  ------      -----
#  1   sport       object
#  2   dstip       object
#  3   dsport      object
#  4   proto       object
#  5   state       object
#  6   service     object
#  7   ct_ftp_cmd  object
#  8   attack_cat  object

########################################### Examine the target column since it is type object 
print(df['attack_cat'].unique())

#### Outputs with similar names grouped together 
# 'No Attack' 
# 'Exploits' 

# 'Reconnaissance' this is not padded with spaces 
# ' Reconnaissance ' this is padded with spaces 
 
# 'DoS' 
# 'Generic' 

# 'Shellcode'
# ' Shellcode ' this is padded with spaces 

# ' Fuzzers' this does not have a space at the end 
#  ' Fuzzers ' this has a space at the end
# 'Worms' 

# 'Backdoors' this has an s at the end
# 'Backdoor'  this does not have an s at the end
# 'Analysis' 

""" Please check this code cell is not mapping them correctly

############################################# Clean the 'attack_cat' column by removing leading/trailing spaces and extra spaces ###########################
df['attack_cat_cleaned'] = df['attack_cat'].str.replace(r'^\s+|\s+$', '', regex=True).replace({
    'No Attack': 'No Attack',
    'Exploits': 'Exploits',
    'Reconnaissance': 'Reconnaissance',
    'DoS': 'DoS',
    'Generic': 'Generic',
    'Shellcode': 'Shellcode',
    'Fuzzers': 'Fuzzers',
    'Worms': 'Worms',
    'Backdoors': 'Backdoor',
    'Analysis': 'Analysis'
})


print(df['attack_cat'].unique())

"""
