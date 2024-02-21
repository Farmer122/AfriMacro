import streamlit as st
from pytrends.request import TrendReq
from datetime import datetime
import wbdata
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Nigeria's GDP Predictions", layout="wide")
st.markdown("""
    <style>
    .main {
       background-color: #0E1117;
    }
    .st-cb, .st-bb {background-color: #0E1117;}
    .st-bb {color: white;}
    .st-at {color: white;}
    .css-145kmo2 {color: white;}
    </style>
    """, unsafe_allow_html=True)

countries = ['NGA']
pycountries = ['NG']

def load_gdp_data(start_year=2019, end_year=2022):
    indicators = {'NY.GDP.MKTP.CD': 'GDP_Current_USD'}
    gdp_data = wbdata.get_dataframe(indicators, country=countries)
    gdp_data.reset_index(inplace=True)
    gdp_data['date'] = pd.to_datetime(gdp_data['date'])
    gdp_data = gdp_data[gdp_data['date'].dt.year.between(start_year, end_year)]
    return gdp_data

def fetch_google_trends(keywords, start_date='2019-01-01', end_date=datetime.today().strftime('%Y-%m-%d')):
    pytrend = TrendReq()
    trends_data_combined = pd.DataFrame()
    for country in pycountries:
        for keyword in keywords:
            pytrend.build_payload(kw_list=[keyword], timeframe=f'{start_date} {end_date}', geo=country)
            data = pytrend.interest_over_time()
            if not data.empty:
                data.reset_index(inplace=True)
                data = data[['date', keyword]]
                if trends_data_combined.empty:
                    trends_data_combined = data
                else:
                    trends_data_combined = trends_data_combined.merge(data, on='date', how='left')
    return trends_data_combined

def preprocess_and_scale_data(trends_data_combined):
    scaler = StandardScaler()
    numeric_columns = trends_data_combined.select_dtypes(include=['float64', 'int64']).columns
    trends_data_combined[numeric_columns] = scaler.fit_transform(trends_data_combined[numeric_columns])
    return trends_data_combined

keywords = ["Economy", "Recession", "Politics", "Unemployment", "Loan", "Interest", "Inflation", "Fintech", "Mobile Payments", "Bitcoin", "News", "JAPA", "Visa"]
gdp_data = load_gdp_data()
trends_data_combined = fetch_google_trends(keywords)
trends_data_preprocessed = preprocess_and_scale_data(trends_data_combined)
gdp_data.sort_values('date', inplace=True)
gdp_dates = pd.to_datetime(gdp_data['date']).map(datetime.toordinal)
gdp_values = gdp_data['GDP_Current_USD'].values
cubic_spline = interp1d(gdp_dates, gdp_values, kind='cubic', bounds_error=False, fill_value="extrapolate")
trends_dates_up_to_2022 = pd.to_datetime(trends_data_preprocessed['date']).map(datetime.toordinal)
trends_data_preprocessed['weekly_gdp'] = cubic_spline(trends_dates_up_to_2022)
trends_data_train = trends_data_preprocessed[trends_data_preprocessed['date'].dt.year <= 2022].copy()
X_train = trends_data_train.drop(columns=['date'] + keywords)
X_train[keywords] = trends_data_train[keywords]
y_train = trends_data_train['weekly_gdp']
network = MLPRegressor(hidden_layer_sizes=(300, 10), solver="adam", activation="relu", learning_rate_init=0.001, tol=1e-4, max_iter=10000, random_state=0)
network.fit(X_train, y_train)
X_all = trends_data_preprocessed.drop(columns=['date'] + keywords)
X_all[keywords] = trends_data_preprocessed[keywords]
y_pred_all = network.predict(X_all)
predictions_df = pd.DataFrame({'date': trends_data_preprocessed['date'], 'weekly_gdp_predictions': y_pred_all})
predictions_df.set_index('date', inplace=True)
monthly_predictions = predictions_df.resample('M').mean()
quarterly_predictions = predictions_df.resample('Q').mean()

prediction_interval = st.sidebar.selectbox("Select Prediction Interval:", ["Weekly", "Monthly", "Quarterly"])
if prediction_interval == "Weekly":
    data_to_plot = predictions_df.reset_index()
elif prediction_interval == "Monthly":
    data_to_plot = monthly_predictions.reset_index()
else:
    data_to_plot = quarterly_predictions.reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(gdp_data['date'], gdp_data['GDP_Current_USD'], label='Actual GDP', color='blue')
if prediction_interval == "Weekly":
    ax.plot(data_to_plot['date'], data_to_plot['weekly_gdp_predictions'], label='Predicted Weekly GDP', color='red', linestyle='--')
elif prediction_interval == "Monthly":
    ax.plot(data_to_plot['date'], data_to_plot['weekly_gdp_predictions'], label='Predicted Monthly GDP', color='green', linestyle='--')
else:
    ax.plot(data_to_plot['date'], data_to_plot['weekly_gdp_predictions'], label='Predicted Quarterly GDP', color='purple', linestyle='--')
ax.set_title(f"Actual vs Predicted GDP ({prediction_interval})")
ax.set_xlabel("Date")
ax.set_ylabel("GDP ($)")
ax.legend()
st.pyplot(fig)
