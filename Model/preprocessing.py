import streamlit as st
from pytrends.request import TrendReq
from datetime import datetime
import wbdata
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import plotly.graph_objs as go



def preprocess_and_scale_data(trends_data_combined):
    scaler = StandardScaler()
    numeric_columns = trends_data_combined.select_dtypes(include=['float64', 'int64']).columns
    trends_data_combined[numeric_columns] = scaler.fit_transform(trends_data_combined[numeric_columns])
    return trends_data_combined
