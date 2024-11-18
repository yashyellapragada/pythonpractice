import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# file with missing values
# (Have taken the noisy version to ignore the disturbce by the induction of noise)
missing_data = pd.read_csv('missingnoisyMichigan.csv')
# Years that have essential values
citation_years = ['2017', '2018', '2019', '2020', '2021', '2022', '2023']

def individual_mean(df):
    df[citation_years] = df[citation_years].apply(lambda row: row.fillna(row.mean()), axis=1)
    return df

def individual_median(df):
    df[citation_years] = df[citation_years].apply(lambda row: row.fillna(row.median()), axis=1)
    return df

def field_mean(df):
    for year in citation_years:
        df[year] = df[year].fillna(df[year].mean())
    return df

def field_median(df):
    for year in citation_years:
        df[year] = df[year].fillna(df[year].median())
    return df

def local_gradient(df):
    df[citation_years] = df[citation_years].apply(lambda row: row.interpolate(method='linear'), axis=1)
    return df

def nearest_neighbor_L1(df):     # Using Manhattan Distance
    filled_df = df.copy() # Starting with the given df, but wil lfill with compputed values
    essential_values = df[citation_years]

    for ind, row in essential_values.iterrows():
        if row.isnull().any():
            complete_data = essential_values.dropna()
            n_neighbors = min(len(complete_data), 1)  # At least 1 neighbor
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric='manhattan').fit(complete_data)
            row_df = pd.DataFrame([row.fillna(0)], columns=citation_years)
            dist, indices = nn.kneighbors(row_df, n_neighbors=n_neighbors) 
            nearest_ind = indices[0][0]
            nearest_faculty = complete_data.iloc[nearest_ind]
            
            row_imputed = row.fillna(nearest_faculty)
            filled_df.loc[ind, citation_years] = row_imputed.values
    return filled_df

def nearest_neighbor_L2(df): # Using Euclidean distance
    filled_df = df.copy()
    essential_values = df[citation_years]

    for ind, row in essential_values.iterrows():
        if row.isnull().any():
            complete_data = essential_values.dropna()
            n_neighbors = min(len(complete_data), 1)  # At least 1 neighbor
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(complete_data)
            row_df = pd.DataFrame([row.fillna(0)], columns=citation_years)
            dist, indices = nn.kneighbors(row_df, n_neighbors=n_neighbors)
            nearest_ind = indices[0][0]
            nearest_faculty = complete_data.iloc[nearest_ind]

            row_imputed = row.fillna(nearest_faculty)
            filled_df.loc[ind, citation_years] = row_imputed.values
    return filled_df

def cal_aae(orig_df, imput_df, miss_df):
    total_error = 0
    total_missing = 0

    for year in citation_years:
        missing_mask = miss_df[year].isna()
        orig_val = orig_df.loc[missing_mask, year].values
        imput_val = imput_df.loc[missing_mask, year].values

        if len(orig_val) > 0:
            # Calculating Average Absolute Error
            error = mean_absolute_error(orig_val, imput_val)
            total_error += error * len(orig_val)
            total_missing += len(orig_val)

    return total_error / total_missing if total_missing > 0 else 0

# All Methods 
methods = {
    'Individual_mean': individual_mean,
    'Individual_median': individual_median,
    'Field_mean': field_mean,
    'Field_median': field_median,
    'Local_gradient': local_gradient,
    'Nearest_neighbor_L1': nearest_neighbor_L1,
    'Nearest_neighbor_L2': nearest_neighbor_L2
}

# file that has original data 
# (Have taken noisy data to ignore the disturbance by induction of noise)
original_data = pd.read_csv('noisyMichigan.csv')

# Empty dictionary to store values from each method
results = {}

# Applying imputation methods and calculating AAE
for m_name, m_func in methods.items():
    imputed_df = m_func(missing_data.copy())
    m_error = cal_aae(original_data, imputed_df, missing_data.copy())
    results[m_name] = m_error

results_df = pd.DataFrame(list(results.items()), columns=['Method', 'Average Absolute Error'])
print(results_df)

# Visualisation
plt.figure(figsize=(10,6))
plt.bar(results_df['Method'], results_df['Average Absolute Error'])
plt.title('Comparison of Imputation Methods')
plt.ylabel('Average Absolute Error')
plt.xticks(rotation=45)
plt.show()
