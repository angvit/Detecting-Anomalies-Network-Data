import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")
st.title("Data Cleaning and Pre-processing")

st.markdown("""
### Identify columns with null values
- `ct_flw_http_mthd`: Filled in the null values.
- `is_ftp_login`: Filled in the null values.
- `attack_cat`: The null values were treated as normal network traffic, so filled in with "Normal".

""")

st.markdown("### Examined the 'attack_cat' (target column)")
st.write("""
Found that there were duplicate values, and values that could be grouped. Before cleaning, there were 14 unique values:

[ nan, 'Exploits', 'Reconnaissance', 'DoS', 'Generic', 'Shellcode', 'Fuzzers', 
'Worms', 'Backdoors', 'Analysis', 'Reconnaissance', 'Backdoor', 'Fuzzers', 
'Shellcode' ] 

Used a lambda function to remove stray spaces, 'Shellcode' (not padded with spaces) and ' Shellcode ' (padded with spaces).
""")

st.markdown("""
### Dropping unnecessary columns
To address an issue with object/hexadecimal outputs, we drop the following columns: `srcip`, `dstip`, `sport`, `dsport`, and `Label`. These columns are removed because they are not necessary for the analysis and were causing issues during processing.
""")

st.markdown("""### Identifying and handling stray values
During the data cleaning process, empty strings were found as stray values in the `ct_ftp_cmd` column. To handle this, we replaced the empty strings with 0 and converted the other values to integers. The following code was used:

```python
df['ct_ftp_cmd'] = df['ct_ftp_cmd'].apply(lambda x: 0 if x == ' ' else int(x))

""")
