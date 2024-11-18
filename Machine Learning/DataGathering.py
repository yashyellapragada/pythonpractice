import numpy as np
import pandas as pd

df = pd.read_csv('Michigan.csv')

def add_noise(df, noise_percentage=0.02, modify_prob=0.3):
    
    df_noisy = df.copy()

    year_columns = ['2017', '2018', '2019', '2020', '2021', '2022', '2023','h-index', 'i-10 index']

    for col in year_columns:
        if col in df_noisy.columns and df_noisy[col].dtype in ['float64', 'int64']:
            df_noisy[col] = df_noisy[col].astype(float)
            modify_mask = np.random.rand(len(df_noisy)) < modify_prob

            noise = np.random.uniform(-noise_percentage, noise_percentage, size=len(df_noisy))
            df_noisy.loc[modify_mask, col] += df_noisy.loc[modify_mask, col] * noise[modify_mask]
            df_noisy[col] = df_noisy[col].astype(int)
    
    return df_noisy

df_noisy = add_noise(df, noise_percentage=0.02)

# Save the noisy DataFrame to a new CSV file with the prefix "noisy"
df_noisy.to_csv('noisyMichigan.csv', index=False)

columns = ['2018','2019','2020','2021']
def add_missingValues(df, columns, modify_prob=0.4, missing_marker='_'):
    df_missingvalues = df.copy()
    modify_mask = np.random.rand(len(df_missingvalues)) < modify_prob

    for idx, row in df_missingvalues[modify_mask].iterrows():
        # Randomly choose one column to omit
        col_to_omit = np.random.choice(columns)
        # Set the chosen column's value to the missing marker
        #df_missingvalues.at[idx, col_to_omit] = missing_marker
        df_missingvalues.at[idx, col_to_omit] = np.nan

    return df_missingvalues

df_missingvalues = add_missingValues(df_noisy, columns)
df_missingvalues.to_csv('missingnoisyMichigan.csv', index=False)