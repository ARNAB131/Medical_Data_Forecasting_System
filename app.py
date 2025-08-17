#!/usr/bin/env python3
"""
Medical Data Forecasting System
Author: Data Scientist
Description: A comprehensive system for forecasting blood pressure and body temperature
             using time series analysis with Streamlit interface
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class MedicalDataGenerator:
    """
    Class for generating realistic medical data (blood pressure and temperature)
    following medical standards and physiological patterns
    """

    def __init__(self, n_samples=100):
        """
        Initialize the medical data generator

        Args:
            n_samples (int): Number of samples to generate
        """
        self.n_samples = n_samples
        self.random_state = 42  # For reproducibility
        np.random.seed(self.random_state)

    def generate_realistic_data(self):
        """
        Generate realistic medical data with temporal patterns

        Returns:
            pd.DataFrame: Generated medical data
        """
        try:
            # Generate timestamps (every 10 minutes for 100 readings ≈ 16.7 hours)
            start_time = datetime.now() - timedelta(hours=17)
            timestamps = [start_time + timedelta(minutes=10 * i) for i in range(self.n_samples)]

            # Generate base trends with circadian rhythm
            time_hours = np.array([(ts.hour + ts.minute / 60) for ts in timestamps])

            # Systolic BP: Normal range 90-140 mmHg with circadian variation
            systolic_base = 120 + 15 * np.sin(2 * np.pi * time_hours / 24 - np.pi / 4)
            systolic_noise = np.random.normal(0, 8, self.n_samples)
            systolic_bp = np.clip(systolic_base + systolic_noise, 85, 180)

            # Diastolic BP: Typically 60-80% of systolic
            diastolic_bp = systolic_bp * (0.65 + 0.1 * np.random.normal(0, 0.1, self.n_samples))
            diastolic_bp = np.clip(diastolic_bp, 50, 110)

            # Body Temperature: Normal range 36.1-37.2°C with circadian rhythm
            temp_base = 36.7 + 0.4 * np.sin(2 * np.pi * time_hours / 24 - np.pi / 3)
            temp_noise = np.random.normal(0, 0.2, self.n_samples)
            body_temp = np.clip(temp_base + temp_noise, 35.5, 38.5)

            # Add some correlation between BP and temperature (mild)
            correlation_factor = 0.3
            systolic_bp += correlation_factor * (body_temp - 36.7) * 2
            diastolic_bp += correlation_factor * (body_temp - 36.7) * 1.5

            # Create DataFrame with proper timestamp formatting
            data = pd.DataFrame({
                'timestamp': timestamps,
                'systolic_bp': np.round(systolic_bp, 1),
                'diastolic_bp': np.round(diastolic_bp, 1),
                'body_temperature': np.round(body_temp, 1)
            })

            # Convert timestamp to string format to avoid Arrow serialization issues
            data['timestamp'] = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

            logger.info(f"Generated {len(data)} medical data samples")
            return data

        except Exception as e:
            logger.error(f"Error generating medical data: {e}")
            raise

    def save_to_csv(self, data, filename="medical_data.csv"):
        """
        Save generated data to CSV file

        Args:
            data (pd.DataFrame): Data to save
            filename (str): Output filename
        """
        try:
            data.to_csv(filename, index=False)
            logger.info(f"Data saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving data to CSV: {e}")
            raise


class MedicalDataForecaster:
    """
    Advanced forecasting class for medical time series data
    Implements multiple forecasting approaches with ensemble methods
    """

    def __init__(self):
        """Initialize the forecasting models"""
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_columns = []

    def create_features(self, data):
        """
        Create time-based features for forecasting

        Args:
            data (pd.DataFrame): Input data with timestamp

        Returns:
            pd.DataFrame: Data with additional features
        """
        try:
            df = data.copy()
            # Ensure timestamp is datetime format for processing
            if df['timestamp'].dtype == 'object':
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['day_of_week'] = df['timestamp'].dt.dayofweek

            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

            # Lag features (previous readings)
            for col in ['systolic_bp', 'diastolic_bp', 'body_temperature']:
                df[f'{col}_lag1'] = df[col].shift(1)
                df[f'{col}_lag2'] = df[col].shift(2)
                df[f'{col}_lag3'] = df[col].shift(3)

            # Rolling statistics
            window_size = 5
            for col in ['systolic_bp', 'diastolic_bp', 'body_temperature']:
                df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
                df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std()

            # Fill NaN values
            df = df.fillna(method='bfill').fillna(method='ffill')

            return df

        except Exception as e:
            logger.error(f"Error creating features: {e}")
            raise

    def prepare_data(self, data, target_cols):
        """
        Prepare data for training and forecasting

        Args:
            data (pd.DataFrame): Input data
            target_cols (list): Target columns to forecast

        Returns:
            tuple: X, y arrays for training
        """
        try:
            # Create features
            df_features = self.create_features(data)

            # Define feature columns (exclude timestamp and targets)
            exclude_cols = ['timestamp'] + target_cols
            self.feature_columns = [col for col in df_features.columns if col not in exclude_cols]

            X = df_features[self.feature_columns].values
            y = df_features[target_cols].values

            # Handle any remaining NaN values
            X = np.nan_to_num(X, nan=0.0)
            y = np.nan_to_num(y, nan=0.0)

            return X, y

        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise

    def train_models(self, data):
        """
        Train forecasting models

        Args:
            data (pd.DataFrame): Training data
        """
        try:
            target_cols = ['systolic_bp', 'diastolic_bp', 'body_temperature']
            X, y = self.prepare_data(data, target_cols)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train models
            self.trained_models = {}
            for name, model in self.models.items():
                self.trained_models[name] = {}
                for i, target in enumerate(target_cols):
                    model_copy = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
                    model_copy.fit(X_scaled, y[:, i])
                    self.trained_models[name][target] = model_copy

            self.is_fitted = True
            logger.info("Models trained successfully")

        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise

    def forecast(self, data, hours_ahead=[3, 6, 9]):
        """
        Generate forecasts for specified time horizons

        Args:
            data (pd.DataFrame): Historical data
            hours_ahead (list): Hours to forecast ahead

        Returns:
            dict: Forecasted values
        """
        try:
            if not self.is_fitted:
                raise ValueError("Models must be trained before forecasting")

            target_cols = ['systolic_bp', 'diastolic_bp', 'body_temperature']

            # Get the last timestamp
            last_timestamp = pd.to_datetime(data['timestamp'].iloc[-1])

            # Create future timestamps
            future_timestamps = []
            for hours in hours_ahead:
                future_time = last_timestamp + timedelta(hours=hours)
                future_timestamps.append(future_time)

            forecasts = {}

            for hours, future_time in zip(hours_ahead, future_timestamps):
                # Create a dummy future record with the same structure
                future_record = data.iloc[-1:].copy()
                future_record['timestamp'] = future_time

                # Combine with historical data to create features
                extended_data = pd.concat([data, future_record], ignore_index=True)
                X_future, _ = self.prepare_data(extended_data, target_cols)
                X_future_scaled = self.scaler.transform(X_future)

                # Use the last record for prediction
                X_pred = X_future_scaled[-1:, :]

                # Ensemble predictions
                predictions = {}
                for target in target_cols:
                    target_preds = []
                    for model_name in self.trained_models:
                        pred = self.trained_models[model_name][target].predict(X_pred)[0]
                        target_preds.append(pred)

                    # Average ensemble
                    predictions[target] = np.mean(target_preds)

                forecasts[f"{hours}h"] = predictions

            logger.info(f"Forecasts generated for {hours_ahead} hours ahead")
            return forecasts

        except Exception as e:
            logger.error(f"Error generating forecasts: {e}")
            raise


class MedicalDataVisualizer:
    """
    Comprehensive visualization class for medical data analysis
    """

    def __init__(self):
        """Initialize the visualizer"""
        self.color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']

    def plot_time_series(self, data):
        """
        Create interactive time series plots

        Args:
            data (pd.DataFrame): Time series data

        Returns:
            plotly figure
        """
        try:
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=['Blood Pressure Over Time', 'Temperature Over Time', 'BP vs Temperature'],
                vertical_spacing=0.08
            )

            # Blood pressure subplot
            fig.add_trace(
                go.Scatter(x=data['timestamp'], y=data['systolic_bp'],
                           name='Systolic BP', line=dict(color='#FF6B6B', width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=data['timestamp'], y=data['diastolic_bp'],
                           name='Diastolic BP', line=dict(color='#4ECDC4', width=2)),
                row=1, col=1
            )

            # Temperature subplot
            fig.add_trace(
                go.Scatter(x=data['timestamp'], y=data['body_temperature'],
                           name='Body Temperature', line=dict(color='#45B7D1', width=2)),
                row=2, col=1
            )

            # Correlation subplot
            fig.add_trace(
                go.Scatter(x=data['body_temperature'], y=data['systolic_bp'],
                           mode='markers', name='Systolic vs Temp',
                           marker=dict(color='#FF6B6B', size=6, opacity=0.6)),
                row=3, col=1
            )

            fig.update_layout(height=800, showlegend=True, title_text="Medical Data Analysis")
            fig.update_xaxes(title_text="Time", row=1, col=1)
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_xaxes(title_text="Body Temperature (°C)", row=3, col=1)
            fig.update_yaxes(title_text="Blood Pressure (mmHg)", row=1, col=1)
            fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
            fig.update_yaxes(title_text="Systolic BP (mmHg)", row=3, col=1)

            return fig

        except Exception as e:
            logger.error(f"Error creating time series plot: {e}")
            raise

    def plot_distribution(self, data):
        """
        Create distribution plots (bell curves)

        Args:
            data (pd.DataFrame): Data for distribution analysis

        Returns:
            plotly figure
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Systolic BP Distribution', 'Diastolic BP Distribution',
                                'Temperature Distribution', 'Combined Statistics']
            )

            # Systolic BP distribution
            fig.add_trace(
                go.Histogram(x=data['systolic_bp'], nbinsx=20, name='Systolic BP',
                             marker_color='#FF6B6B', opacity=0.7, histnorm='probability density'),
                row=1, col=1
            )

            # Diastolic BP distribution
            fig.add_trace(
                go.Histogram(x=data['diastolic_bp'], nbinsx=20, name='Diastolic BP',
                             marker_color='#4ECDC4', opacity=0.7, histnorm='probability density'),
                row=1, col=2
            )

            # Temperature distribution
            fig.add_trace(
                go.Histogram(x=data['body_temperature'], nbinsx=20, name='Temperature',
                             marker_color='#45B7D1', opacity=0.7, histnorm='probability density'),
                row=2, col=1
            )

            # Statistics box plot
            fig.add_trace(
                go.Box(y=data['systolic_bp'], name='Systolic', marker_color='#FF6B6B'),
                row=2, col=2
            )
            fig.add_trace(
                go.Box(y=data['diastolic_bp'], name='Diastolic', marker_color='#4ECDC4'),
                row=2, col=2
            )
            fig.add_trace(
                go.Box(y=data['body_temperature'], name='Temperature', marker_color='#45B7D1'),
                row=2, col=2
            )

            fig.update_layout(height=600, showlegend=True, title_text="Data Distributions")

            return fig

        except Exception as e:
            logger.error(f"Error creating distribution plots: {e}")
            raise

    def create_correlation_heatmap(self, data):
        """
        Create correlation heatmap

        Args:
            data (pd.DataFrame): Data for correlation analysis

        Returns:
            plotly figure
        """
        try:
            # Calculate correlation matrix
            numeric_cols = ['systolic_bp', 'diastolic_bp', 'body_temperature']
            corr_matrix = data[numeric_cols].corr()

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 12},
                hoverongaps=False
            ))

            fig.update_layout(
                title="Correlation Matrix of Medical Parameters",
                height=400
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            raise


def main():
    """
    Main Streamlit application
    """
    st.set_page_config(
        page_title="Medical Data Forecasting System",
        page_icon="🏥",
        layout="wide"
    )

    st.title("🏥 Medical Data Forecasting System")
    st.markdown("### Advanced Blood Pressure and Temperature Analysis & Forecasting")

    # Sidebar controls
    st.sidebar.header("Configuration")

    # Generate or upload data option
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["Generate synthetic data", "Upload CSV file"]
    )

    try:
        if data_option == "Generate synthetic data":
            n_samples = st.sidebar.slider("Number of samples", 50, 200, 100)

            if st.sidebar.button("Generate New Data"):
                with st.spinner("Generating medical data..."):
                    generator = MedicalDataGenerator(n_samples)
                    data = generator.generate_realistic_data()
                    st.session_state.data = data
                    st.success(f"Generated {len(data)} data points successfully!")

        else:
            uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                # Ensure timestamp column is properly formatted for Arrow
                if 'timestamp' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                st.session_state.data = data
                st.success("Data uploaded successfully!")

        # Initialize default data if none exists
        if 'data' not in st.session_state:
            generator = MedicalDataGenerator(100)
            st.session_state.data = generator.generate_realistic_data()

        data = st.session_state.data

        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Readings", len(data))
        with col2:
            st.metric("Avg Systolic BP", f"{data['systolic_bp'].mean():.1f} mmHg")
        with col3:
            st.metric("Avg Diastolic BP", f"{data['diastolic_bp'].mean():.1f} mmHg")
        with col4:
            st.metric("Avg Temperature", f"{data['body_temperature'].mean():.1f}°C")

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Analysis", "🔮 Forecasting", "📈 Visualizations", "📋 Raw Data"])

        with tab1:
            st.subheader("Statistical Analysis")

            visualizer = MedicalDataVisualizer()

            # Distribution plots
            st.plotly_chart(visualizer.plot_distribution(data), use_container_width=True)

            # Correlation heatmap
            st.plotly_chart(visualizer.create_correlation_heatmap(data), use_container_width=True)

            # Statistical summary
            st.subheader("Statistical Summary")
            st.dataframe(data.describe())

        with tab2:
            st.subheader("Medical Parameter Forecasting")

            if st.button("Generate Forecasts", type="primary"):
                with st.spinner("Training models and generating forecasts..."):
                    forecaster = MedicalDataForecaster()
                    forecaster.train_models(data)
                    forecasts = forecaster.forecast(data)

                    st.success("Forecasts generated successfully!")

                    # Display forecasts
                    col1, col2, col3 = st.columns(3)

                    for i, (time_horizon, predictions) in enumerate(forecasts.items()):
                        col = [col1, col2, col3][i]
                        with col:
                            st.markdown(f"**{time_horizon} Forecast**")
                            st.metric("Systolic BP", f"{predictions['systolic_bp']:.1f} mmHg")
                            st.metric("Diastolic BP", f"{predictions['diastolic_bp']:.1f} mmHg")
                            st.metric("Temperature", f"{predictions['body_temperature']:.1f}°C")

                    # Store forecasts in session state
                    st.session_state.forecasts = forecasts

        with tab3:
            st.subheader("Interactive Visualizations")

            visualizer = MedicalDataVisualizer()

            # Time series plot
            st.plotly_chart(visualizer.plot_time_series(data), use_container_width=True)

            # Additional charts
            col1, col2 = st.columns(2)

            with col1:
                # Pie chart for BP categories
                bp_categories = pd.cut(data['systolic_bp'],
                                       bins=[0, 120, 140, 180, float('inf')],
                                       labels=['Normal', 'Elevated', 'High', 'Very High'])
                bp_counts = bp_categories.value_counts()

                fig_pie = px.pie(values=bp_counts.values, names=bp_counts.index,
                                 title="Blood Pressure Categories Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                # Temperature ranges
                temp_ranges = pd.cut(data['body_temperature'],
                                     bins=[35, 36.5, 37.2, 38, float('inf')],
                                     labels=['Low', 'Normal', 'Elevated', 'Fever'])
                temp_counts = temp_ranges.value_counts()

                fig_temp = px.bar(x=temp_counts.index, y=temp_counts.values,
                                  title="Temperature Range Distribution",
                                  color=temp_counts.values,
                                  color_continuous_scale="Viridis")
                st.plotly_chart(fig_temp, use_container_width=True)

        with tab4:
            st.subheader("Raw Data View")
            st.dataframe(data, use_container_width=True)

            # Download button
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f"medical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Application error: {e}")


if __name__ == "__main__":
    main()