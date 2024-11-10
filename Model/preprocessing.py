import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_and_scale_data(trends_data_combined):
    scaler = StandardScaler()
    numeric_columns = trends_data_combined.select_dtypes(include=['float64', 'int64']).columns
    trends_data_combined[numeric_columns] = scaler.fit_transform(trends_data_combined[numeric_columns])
    return trends_data_combined
