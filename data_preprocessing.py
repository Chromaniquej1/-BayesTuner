import pandas as pd
from scipy.stats import zscore

def load_and_preprocess_data():
    # Read the data set
    df = pd.read_csv(
        "https://data.heatonresearch.com/data/t81-558/jh-simple-dataset.csv",
        na_values=['NA', '?'])

    # Generate dummies for job and area
    df = pd.concat([df, pd.get_dummies(df['job'], prefix="job", dtype=int)], axis=1)
    df.drop('job', axis=1, inplace=True)
    
    df = pd.concat([df, pd.get_dummies(df['area'], prefix="area", dtype=int)], axis=1)
    df.drop('area', axis=1, inplace=True)
    
    # Missing values for income
    df['income'] = df['income'].fillna(df['income'].median())

    # Standardize ranges
    df['income'] = zscore(df['income'])
    df['aspect'] = zscore(df['aspect'])
    df['save_rate'] = zscore(df['save_rate'])
    df['age'] = zscore(df['age'])
    df['subscriptions'] = zscore(df['subscriptions'])

    # Convert to numpy - Classification
    x_columns = df.columns.drop('product').drop('id')
    x = df[x_columns].values
    dummies = pd.get_dummies(df['product'])  # Classification
    y = dummies.values

    return x, y
