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

st.set_page_config(page_title="Nigeria's GDP Predictions", layout="wide")
st.markdown("""
    <style>
    .main {
       background-color: #010100;
    }
    .st-cb, .st-bb {background-color: #010100;}
    .st-bb {color: #fafafb;}
    .st-at {color: #fafafb;}
    .css-145kmo2 {color: #fafafb;}
    </style>
    """, unsafe_allow_html=True)

countries = ['NGA']
pycountries = ['NG']

st.title('Nigeria GDP Nowcast')
st.markdown('This project uses Machine Learning and High Frequency Data (google trends) to make GDP Nowcasts for Nigeria')
st.markdown('Author: Jamal Lawal')

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

def create_ensemble_model():
    models = [
        ('MLP', MLPRegressor(hidden_layer_sizes=(300, 10), solver="adam", activation="relu", learning_rate_init=0.001, max_iter=10000, random_state=42)),
        ('RF', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('GB', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('SVR', SVR(kernel='rbf'))
    ]
    return models

def train_ensemble(X, y, models):
    for name, model in models:
        model.fit(X, y)
    return models

def predict_ensemble(X, models):
    predictions = []
    for name, model in models:
        pred = model.predict(X)
        predictions.append(pred)
    return np.mean(predictions, axis=0)

# Main code
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

# Prepare data for ensemble
X = trends_data_preprocessed.drop(columns=['date', 'weekly_gdp'] + keywords)
X[keywords] = trends_data_preprocessed[keywords]
y = trends_data_preprocessed['weekly_gdp']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train the ensemble
models = create_ensemble_model()
trained_models = train_ensemble(X_scaled, y, models)

# Make predictions using the ensemble
ensemble_predictions = predict_ensemble(X_scaled, trained_models)

# Create predictions dataframe
predictions_df = pd.DataFrame({
    'date': trends_data_preprocessed['date'],
    'weekly_gdp_predictions': ensemble_predictions
})
predictions_df.set_index('date', inplace=True)

# Resample to monthly and quarterly predictions
monthly_predictions = predictions_df.resample('M').mean()
quarterly_predictions = predictions_df.resample('Q').mean()

# Evaluate the ensemble model using time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
mse_scores = []

for train_index, test_index in tscv.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    trained_models = train_ensemble(X_train, y_train, models)
    ensemble_pred = predict_ensemble(X_test, trained_models)
    
    mse = mean_squared_error(y_test, ensemble_pred)
    mse_scores.append(mse)

average_mse = np.mean(mse_scores)
st.write(f"Average MSE across folds: {average_mse}")

# Streamlit UI components
prediction_interval = st.sidebar.selectbox("Select Prediction Interval:", ["Weekly", "Monthly", "Quarterly"])

# Prepare the data to plot based on the selected prediction interval
if prediction_interval == "Weekly":
    data_to_plot = predictions_df.reset_index()
elif prediction_interval == "Monthly":
    data_to_plot = monthly_predictions.reset_index()
else:  # Quarterly
    data_to_plot = quarterly_predictions.reset_index()

# Initialize a Plotly figure
fig = go.Figure()

# Add the actual GDP data trace
fig.add_trace(go.Scatter(x=gdp_data['date'], y=gdp_data['GDP_Current_USD'],
                         mode='lines', name='Actual GDP',
                         line=dict(color='blue')))

# Add the predicted GDP data trace based on the selected interval
fig.add_trace(go.Scatter(x=data_to_plot['date'], y=data_to_plot['weekly_gdp_predictions'],
                         mode='lines+markers', name=f'Ensemble Predicted GDP ({prediction_interval})',
                         line=dict(color='red', dash='dash')))

# Update the layout to customize the appearance
fig.update_layout(
    title=f'Actual vs Ensemble Predicted GDP ({prediction_interval})',
    xaxis_title='Date',
    yaxis_title='GDP ($)',
    template="plotly_dark",
    plot_bgcolor='rgba(0, 0, 0, 0)',
    xaxis=dict(
        showline=True,
        showgrid=True,
        gridcolor='grey',
        linecolor='grey',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='white',
        ),
    ),
    yaxis=dict(
        showline=True,
        showgrid=True,
        gridcolor='grey',
        linecolor='grey',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='white',
        ),
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig, use_container_width=True)
