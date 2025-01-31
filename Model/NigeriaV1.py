import streamlit as st
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from Model.data_loader import load_gdp_data, fetch_google_trends
from Model.preprocessing import preprocess_and_scale_data
from Model.modelling import create_ensemble_model, train_ensemble, predict_ensemble

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


# Main code
keywords = ["economics", "fuel", "land", "market", "job", "Loan", "credit", "price", "payment", "Bitcoin", "japa", "visa"]
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
