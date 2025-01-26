import streamlit as st
from datetime import datetime, timedelta
import os
import pickle

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
st.markdown(
    """
    <style>
    .main {
       background-color: #010100;
    }
    .st-cb, .st-bb {background-color: #010100;}
    .st-bb {color: #fafafb;}
    .st-at {color: #fafafb;}
    .css-145kmo2 {color: #fafafb;}
    </style>
    """,
    unsafe_allow_html=True
)


st.title('Nigeria GDP Nowcast')
st.markdown('This project uses Machine Learning and High Frequency Data (Google Trends) to make GDP Nowcasts for Nigeria')
st.markdown('Author: Jamal Lawal')


# ----------------------------------
# 1. HELPER FUNCTIONS FOR CACHING
# ----------------------------------
def save_cache(data_dict, filename="predictions_cache.pkl"):
    """Saves a dictionary of data to a local pickle file."""
    with open(filename, "wb") as f:
        pickle.dump(data_dict, f)

def load_cache(filename="predictions_cache.pkl"):
    """Loads cached data from a local pickle file if it exists."""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

def is_cache_valid(cached_data, max_age_days=7):
    """Checks if the cached data is newer than `max_age_days`."""
    if not cached_data or "last_updated" not in cached_data:
        return False
    last_updated = cached_data["last_updated"]
    return (datetime.now() - last_updated) < timedelta(days=max_age_days)


# ------------------------------------------------
# 2. FUNCTION TO GENERATE PREDICTIONS & STATISTICS
# ------------------------------------------------
def generate_predictions_and_metrics():
    """
    Generates the model predictions, time-series cross-validation MSE,
    and returns them along with the current timestamp.
    """
    # Data loading
    keywords = ["economics", "fuel", "land", "market", "job", "Loan", "credit", "price", "payment", "Bitcoin", "japa", "visa"]
    gdp_data = load_gdp_data()
    trends_data_combined = fetch_google_trends(keywords)
    trends_data_preprocessed = preprocess_and_scale_data(trends_data_combined)

    # Sort and interpolate GDP data
    gdp_data.sort_values('date', inplace=True)
    gdp_dates = pd.to_datetime(gdp_data['date']).map(datetime.toordinal)
    gdp_values = gdp_data['GDP_Current_USD'].values

    cubic_spline = interp1d(gdp_dates, gdp_values, kind='cubic',
                            bounds_error=False, fill_value="extrapolate")

    # Apply spline interpolation to high frequency data
    trends_dates = pd.to_datetime(trends_data_preprocessed['date']).map(datetime.toordinal)
    trends_data_preprocessed['weekly_gdp'] = cubic_spline(trends_dates)

    # Prepare data for ensemble
    X = trends_data_preprocessed.drop(columns=['date', 'weekly_gdp'] + keywords)
    X[keywords] = trends_data_preprocessed[keywords]
    y = trends_data_preprocessed['weekly_gdp']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create and train the ensemble
    models = create_ensemble_model()
    trained_models = train_ensemble(X_scaled, y, models)

    # Make predictions
    ensemble_predictions = predict_ensemble(X_scaled, trained_models)

    # Store predictions in a DataFrame
    predictions_df = pd.DataFrame({
        'date': trends_data_preprocessed['date'],
        'weekly_gdp_predictions': ensemble_predictions
    })
    predictions_df.set_index('date', inplace=True)

    # Evaluate MSE via Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)
    mse_scores = []
    for train_index, test_index in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        trained_models_fold = train_ensemble(X_train, y_train, models)
        ensemble_pred_fold = predict_ensemble(X_test, trained_models_fold)
        mse = mean_squared_error(y_test, ensemble_pred_fold)
        mse_scores.append(mse)

    average_mse = np.mean(mse_scores)

    # Return everything needed
    return {
        "predictions_df": predictions_df,
        "gdp_data": gdp_data,
        "average_mse": average_mse,
        "last_updated": datetime.now()  # to track cache time
    }


# ---------------------------------------------
# 3. MAIN LOGIC: CHECK CACHE OR GENERATE FRESH
# ---------------------------------------------
cached_data = load_cache()

if is_cache_valid(cached_data, max_age_days=7):
    st.write("Using cached predictions (less than 7 days old).")
    predictions_df = cached_data["predictions_df"]
    gdp_data = cached_data["gdp_data"]
    average_mse = cached_data["average_mse"]
else:
    st.write("Cache is missing or expired. Generating new predictions...")
    fresh_data = generate_predictions_and_metrics()
    predictions_df = fresh_data["predictions_df"]
    gdp_data = fresh_data["gdp_data"]
    average_mse = fresh_data["average_mse"]
    # Save to cache
    save_cache(fresh_data)

st.write(f"Average MSE across folds: {average_mse}")


# ----------------------------------------
# 4. STREAMLIT UI FOR PLOTTING PREDICTIONS
# ----------------------------------------
# Resample to monthly and quarterly
monthly_predictions = predictions_df.resample('ME').mean()
quarterly_predictions = predictions_df.resample('QE').mean()

# Choose interval from sidebar
prediction_interval = st.sidebar.selectbox("Select Prediction Interval:", ["Weekly", "Monthly", "Quarterly"])

# Prepare data based on the selected interval
if prediction_interval == "Weekly":
    data_to_plot = predictions_df.reset_index()
elif prediction_interval == "Monthly":
    data_to_plot = monthly_predictions.reset_index()
else:  # Quarterly
    data_to_plot = quarterly_predictions.reset_index()

# Plotly figure
fig = go.Figure()

# Actual GDP trace
fig.add_trace(
    go.Scatter(
        x=gdp_data['date'],
        y=gdp_data['GDP_Current_USD'],
        mode='lines',
        name='Actual GDP',
        line=dict(color='blue')
    )
)

# Predicted GDP trace
fig.add_trace(
    go.Scatter(
        x=data_to_plot['date'],
        y=data_to_plot['weekly_gdp_predictions'],
        mode='lines+markers',
        name=f'Ensemble Predicted GDP ({prediction_interval})',
        line=dict(color='red', dash='dash')
    )
)

# Figure layout
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
        tickfont=dict(family='Arial', size=12, color='white'),
    ),
    yaxis=dict(
        showline=True,
        showgrid=True,
        gridcolor='grey',
        linecolor='grey',
        linewidth=2,
        ticks='outside',
        tickfont=dict(family='Arial', size=12, color='white'),
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig, use_container_width=True)
