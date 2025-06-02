import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    GRU,
    Input,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from prophet import Prophet


class WalmartForecastingModels:
    """Forecasting models optimized for Walmart competition"""

    def __init__(self, data):
        self.data = data
        self.models = {}
        self.results = {}
        self.feature_columns = []
        self.train_data = None
        self.val_data = None

    def prepare_walmart_data(self, validation_weeks=8):
        """Prepare data specifically for Walmart forecasting with holiday weights"""
        print("=== PREPARING WALMART DATA FOR MODELING ===")

        # Remove rows without sales data (early periods with insufficient lags)
        self.data_clean = self.data.dropna(subset=["Weekly_Sales"])

        # Sort by date for proper time series split
        self.data_clean = self.data_clean.sort_values(["Store", "Dept", "Date"])

        # Select feature columns
        exclude_cols = ["Weekly_Sales", "Store", "Dept", "Date", "Type"]
        self.feature_columns = [
            col
            for col in self.data_clean.columns
            if col not in exclude_cols and not col.endswith("_scaled")
        ]

        # Time-based split (last N weeks for validation)
        unique_dates = sorted(self.data_clean["Date"].unique())
        split_date = unique_dates[-validation_weeks]

        self.train_data = self.data_clean[self.data_clean["Date"] < split_date].copy()
        self.val_data = self.data_clean[self.data_clean["Date"] >= split_date].copy()

        # Create holiday weights for training data
        self.train_weights = self.train_data["Holiday_Weight"].values
        self.val_weights = self.val_data["Holiday_Weight"].values

        print(f"Training data: {self.train_data.shape}")
        print(f"Validation data: {self.val_data.shape}")
        print(f"Features: {len(self.feature_columns)}")
        print(f"Holiday weeks in training: {(self.train_weights == 5.0).sum()}")

        return self.train_data, self.val_data

    def prophet_walmart_model(self):
        """Prophet model optimized for Walmart data"""
        print("=== TRAINING WALMART-OPTIMIZED PROPHET MODEL ===")
        start_time = time.time()

        try:
            # Aggregate by date for Prophet (total sales across all stores/depts)
            prophet_train = (
                self.train_data.groupby("Date")
                .agg(
                    {
                        "Weekly_Sales": "sum",
                        "Temperature": "mean",
                        "Fuel_Price": "mean",
                        "CPI": "mean",
                        "Unemployment": "mean",
                        "IsHoliday": "max",
                        "Total_MarkDown": "sum",
                    }
                )
                .reset_index()
            )

            prophet_train.columns = [
                "ds",
                "y",
                "temperature",
                "fuel_price",
                "cpi",
                "unemployment",
                "holiday",
                "markdown",
            ]

            # Initialize Prophet with Walmart-specific settings
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,  # Weekly data, not daily
                daily_seasonality=False,
                seasonality_mode="multiplicative",
                changepoint_prior_scale=0.1,  # More flexible for retail
                seasonality_prior_scale=15,  # Strong seasonality in retail
                holidays_prior_scale=25,  # Strong holiday effects
                interval_width=0.95,
            )

            # Add regressors
            regressors = [
                "temperature",
                "fuel_price",
                "cpi",
                "unemployment",
                "markdown",
            ]
            for regressor in regressors:
                model.add_regressor(regressor)

            # Add custom holidays (major retail holidays)
            holidays = pd.DataFrame(
                {
                    "holiday": ["Super Bowl", "Labor Day", "Thanksgiving", "Christmas"]
                    * 3,
                    "ds": [
                        "2010-02-12",
                        "2010-09-10",
                        "2010-11-26",
                        "2010-12-31",
                        "2011-02-11",
                        "2011-09-09",
                        "2011-11-25",
                        "2011-12-30",
                        "2012-02-10",
                        "2012-09-07",
                        "2012-11-23",
                        "2012-12-28",
                    ],
                }
            )
            holidays["ds"] = pd.to_datetime(holidays["ds"])
            model.holidays = holidays

            # Fit model
            model.fit(prophet_train)

            # Prepare validation data
            prophet_val = (
                self.val_data.groupby("Date")
                .agg(
                    {
                        "Temperature": "mean",
                        "Fuel_Price": "mean",
                        "CPI": "mean",
                        "Unemployment": "mean",
                        "IsHoliday": "max",
                        "Total_MarkDown": "sum",
                        "Weekly_Sales": "sum",
                    }
                )
                .reset_index()
            )

            prophet_val.columns = [
                "ds",
                "temperature",
                "fuel_price",
                "cpi",
                "unemployment",
                "holiday",
                "markdown",
                "actual",
            ]

            # Make predictions
            forecast = model.predict(prophet_val[["ds"] + regressors])

            # Extract predictions
            predictions = forecast["yhat"].values
            actual = prophet_val["actual"].values

            training_time = time.time() - start_time

            self.models["Prophet"] = model
            self.results["Prophet"] = {
                "predictions": predictions,
                "actual": actual,
                "training_time": training_time,
                "model_type": "Statistical",
                "forecast": forecast,
                "weights": np.ones(
                    len(predictions)
                ),  # Simplified weights for aggregated data
            }

            print(f"Prophet model trained in {training_time:.2f} seconds")
            return model, predictions

        except Exception as e:
            print(f"Prophet training failed: {e}")
            return None, None

    def lstm_walmart_model(self, sequence_length=5, epochs=50):
        """LSTM model with holiday weighting for Walmart data"""
        print("=== TRAINING WALMART LSTM MODEL ===")
        start_time = time.time()

        try:
            # Select top features for LSTM
            feature_importance = self._calculate_feature_importance()
            top_features = feature_importance.head(15).index.tolist()

            # Ensure critical features are included
            critical_features = [
                "IsHoliday",
                "Total_MarkDown",
                "Month_sin",
                "Month_cos",
                "Sales_lag_1",
                "Sales_rolling_mean_4",
            ]
            for feat in critical_features:
                if feat in self.train_data.columns and feat not in top_features:
                    top_features.append(feat)

            # Create sequences for time series
            def create_sequences_walmart(
                data, seq_length, features, target="Weekly_Sales"
            ):
                X, y, weights = [], [], []

                # Process by store-department combinations
                for (store, dept), group in data.groupby(["Store", "Dept"]):
                    if len(group) < seq_length + 1:
                        continue

                    group_sorted = group.sort_values("Date")

                    for i in range(seq_length, len(group_sorted)):
                        # Features sequence
                        X.append(group_sorted[features].iloc[i - seq_length : i].values)
                        # Target
                        y.append(group_sorted[target].iloc[i])
                        # Holiday weight
                        weights.append(group_sorted["Holiday_Weight"].iloc[i])

                return np.array(X), np.array(y), np.array(weights)

            # Create training sequences
            X_train, y_train, train_weights = create_sequences_walmart(
                self.train_data, sequence_length, top_features
            )

            # Create validation sequences
            X_val, y_val, val_weights = create_sequences_walmart(
                self.val_data, sequence_length, top_features
            )

            if len(X_train) == 0 or len(X_val) == 0:
                print("Insufficient data for LSTM sequences")
                return None, None

            # Scale features
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_train_scaled = scaler_X.fit_transform(
                X_train.reshape(-1, X_train.shape[-1])
            ).reshape(X_train.shape)

            X_val_scaled = scaler_X.transform(
                X_val.reshape(-1, X_val.shape[-1])
            ).reshape(X_val.shape)

            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

            # Build LSTM with holiday-aware architecture
            model = Sequential(
                [
                    LSTM(
                        64,
                        return_sequences=True,
                        input_shape=(sequence_length, len(top_features)),
                    ),
                    Dropout(0.3),
                    LSTM(32, return_sequences=True),
                    Dropout(0.2),
                    LSTM(16, return_sequences=False),
                    Dropout(0.2),
                    Dense(32, activation="relu"),
                    Dense(16, activation="relu"),
                    Dense(1),
                ]
            )

            model.compile(
                optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"]
            )

            # Train with callbacks
            callbacks = [
                EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
                ),
            ]

            history = model.fit(
                X_train_scaled,
                y_train_scaled,
                validation_data=(
                    X_val_scaled,
                    scaler_y.transform(y_val.reshape(-1, 1)).flatten(),
                ),
                epochs=epochs,
                batch_size=64,
                callbacks=callbacks,
                verbose=0,
            )

            # Make predictions
            y_pred_scaled = model.predict(X_val_scaled, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

            training_time = time.time() - start_time

            self.models["LSTM"] = model
            self.results["LSTM"] = {
                "predictions": y_pred,
                "actual": y_val,
                "weights": val_weights,
                "training_time": training_time,
                "model_type": "Neural Network",
                "history": history,
                "scalers": {"X": scaler_X, "y": scaler_y},
                "features": top_features,
            }

            print(f"LSTM model trained in {training_time:.2f} seconds")
            print(f"Using {len(top_features)} features")
            return model, y_pred

        except Exception as e:
            print(f"LSTM training failed: {e}")
            return None, None

    def transformer_walmart_model(self, sequence_length=5, epochs=40):
        """Transformer model for Walmart time series"""
        print("=== TRAINING WALMART TRANSFORMER MODEL ===")
        start_time = time.time()

        try:
            # Use aggregated data for transformer (computational efficiency)
            agg_train = (
                self.train_data.groupby(["Store", "Date"])
                .agg(
                    {
                        "Weekly_Sales": "sum",
                        "Temperature": "mean",
                        "Fuel_Price": "mean",
                        "Unemployment": "mean",
                        "IsHoliday": "max",
                        "Total_MarkDown": "sum",
                        "Holiday_Weight": "max",
                    }
                )
                .reset_index()
            )

            agg_val = (
                self.val_data.groupby(["Store", "Date"])
                .agg(
                    {
                        "Weekly_Sales": "sum",
                        "Temperature": "mean",
                        "Fuel_Price": "mean",
                        "Unemployment": "mean",
                        "IsHoliday": "max",
                        "Total_MarkDown": "sum",
                        "Holiday_Weight": "max",
                    }
                )
                .reset_index()
            )

            # Prepare features
            features = [
                "Temperature",
                "Fuel_Price",
                "Unemployment",
                "IsHoliday",
                "Total_MarkDown",
            ]

            # Create sequences
            def create_transformer_sequences(
                data, seq_len, features, target="Weekly_Sales"
            ):
                X, y, weights = [], [], []

                for store in data["Store"].unique():
                    store_data = data[data["Store"] == store].sort_values("Date")

                    if len(store_data) < seq_len + 1:
                        continue

                    for i in range(seq_len, len(store_data)):
                        X.append(store_data[features].iloc[i - seq_len : i].values)
                        y.append(store_data[target].iloc[i])
                        weights.append(store_data["Holiday_Weight"].iloc[i])

                return np.array(X), np.array(y), np.array(weights)

            X_train, y_train, train_weights = create_transformer_sequences(
                agg_train, sequence_length, features
            )
            X_val, y_val, val_weights = create_transformer_sequences(
                agg_val, sequence_length, features
            )

            if len(X_train) == 0 or len(X_val) == 0:
                print("Insufficient data for Transformer")
                return None, None

            # Scale data
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_train_scaled = scaler_X.fit_transform(
                X_train.reshape(-1, X_train.shape[-1])
            ).reshape(X_train.shape)

            X_val_scaled = scaler_X.transform(
                X_val.reshape(-1, X_val.shape[-1])
            ).reshape(X_val.shape)

            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

            # Build Transformer model
            inputs = Input(shape=(sequence_length, len(features)))

            # Multi-head attention
            attention_output = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(
                inputs, inputs
            )

            # Add & Norm
            attention_output = LayerNormalization()(inputs + attention_output)

            # Feed Forward Network
            ffn_output = Dense(128, activation="relu")(attention_output)
            ffn_output = Dropout(0.2)(ffn_output)
            ffn_output = Dense(len(features))(ffn_output)

            # Add & Norm
            ffn_output = LayerNormalization()(attention_output + ffn_output)

            # Global pooling and output
            pooled = GlobalAveragePooling1D()(ffn_output)
            pooled = Dense(64, activation="relu")(pooled)
            pooled = Dropout(0.3)(pooled)
            outputs = Dense(1)(pooled)

            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"]
            )

            # Train model
            callbacks = [
                EarlyStopping(
                    monitor="val_loss", patience=8, restore_best_weights=True
                ),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
            ]

            history = model.fit(
                X_train_scaled,
                y_train_scaled,
                validation_data=(
                    X_val_scaled,
                    scaler_y.transform(y_val.reshape(-1, 1)).flatten(),
                ),
                epochs=epochs,
                batch_size=32,
                callbacks=callbacks,
                verbose=0,
            )

            # Make predictions
            y_pred_scaled = model.predict(X_val_scaled, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

            training_time = time.time() - start_time

            self.models["Transformer"] = model
            self.results["Transformer"] = {
                "predictions": y_pred,
                "actual": y_val,
                "weights": val_weights,
                "training_time": training_time,
                "model_type": "Neural Network",
                "history": history,
            }

            print(f"Transformer model trained in {training_time:.2f} seconds")
            return model, y_pred

        except Exception as e:
            print(f"Transformer training failed: {e}")
            return None, None

    def random_forest_walmart_model(self):
        """Random Forest baseline with Walmart-specific features"""
        print("=== TRAINING WALMART RANDOM FOREST MODEL ===")
        start_time = time.time()

        try:
            # Prepare features
            feature_cols = [
                col for col in self.feature_columns if col in self.train_data.columns
            ][
                :20
            ]  # Top 20 features

            X_train = self.train_data[feature_cols].fillna(0)
            y_train = self.train_data["Weekly_Sales"]
            X_val = self.val_data[feature_cols].fillna(0)
            y_val = self.val_data["Weekly_Sales"]

            # Train Random Forest with holiday sample weights
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )

            # Fit with sample weights (holiday weeks weighted 5x)
            sample_weights = self.train_data["Holiday_Weight"].values
            model.fit(X_train, y_train, sample_weight=sample_weights)

            # Make predictions
            y_pred = model.predict(X_val)

            training_time = time.time() - start_time

            # Feature importance
            feature_importance = pd.DataFrame(
                {"feature": feature_cols, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)

            self.models["RandomForest"] = model
            self.results["RandomForest"] = {
                "predictions": y_pred,
                "actual": y_val.values,
                "weights": self.val_data["Holiday_Weight"].values,
                "training_time": training_time,
                "model_type": "Tree-based",
                "feature_importance": feature_importance,
            }

            print(f"Random Forest model trained in {training_time:.2f} seconds")
            print("Top 5 most important features:")
            print(feature_importance.head())

            return model, y_pred

        except Exception as e:
            print(f"Random Forest training failed: {e}")
            return None, None

    def _calculate_feature_importance(self):
        """Calculate feature importance using correlation and variance"""
        numeric_features = self.train_data.select_dtypes(include=[np.number]).columns
        numeric_features = [
            col
            for col in numeric_features
            if col not in ["Store", "Dept", "Weekly_Sales"]
        ]

        # Calculate correlation with target
        correlations = {}
        for feature in numeric_features:
            if feature in self.train_data.columns:
                corr = abs(
                    self.train_data[feature].corr(self.train_data["Weekly_Sales"])
                )
                correlations[feature] = corr if not np.isnan(corr) else 0

        return pd.Series(correlations).sort_values(ascending=False)


if __name__ == "__main__":
    from src.data_loader import WalmartDataLoader
    from src.data_processing import WalmartComprehensiveEDA
    from src.feature_engineering import WalmartFeatureEngineering

    data_loader = WalmartDataLoader()
    data_loader.load_data()
    train_data = data_loader.train_data
    test_data = data_loader.test_data
    features_data = data_loader.features_data
    stores_data = data_loader.stores_data

    eda = WalmartComprehensiveEDA(train_data, test_data, features_data, stores_data)
    merged_data = eda.merge_datasets()

    feature_eng = WalmartFeatureEngineering(merged_data)
    processed_data = feature_eng.create_walmart_features()
    processed_data = feature_eng.handle_missing_values()
    print("Feature Engineering class ready!")

    forecasting_models = WalmartForecastingModels(processed_data)
    train_data, val_data = forecasting_models.prepare_walmart_data()

    # models
    prophet_model, prophet_pred = forecasting_models.prophet_walmart_model()
    lstm_model, lstm_pred = forecasting_models.lstm_walmart_model()
    rf_model, rf_pred = forecasting_models.random_forest_walmart_model()
    trans_model, trans_pred = forecasting_models.transformer_walmart_model()

    print("Forecasting Models class ready!")
