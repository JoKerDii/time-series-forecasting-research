import pandas as pd
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

# Core libraries
from sklearn.preprocessing import StandardScaler

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    GRU,
    Input,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D,
    Conv1D,
    MaxPooling1D,
    Concatenate,
    BatchNormalization,
    Add,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2

# Statistical models
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Advanced time series
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available. Install with: pip install torch")
    TORCH_AVAILABLE = False


class AdvancedWalmartForecastingModels:
    """Advanced forecasting models for Walmart competition including hierarchical and causal approaches"""

    def __init__(self, data):
        self.data = data
        self.models = {}
        self.results = {}
        self.hierarchical_structure = None
        self.causal_graph = None
        self.feature_columns = []
        self.train_data = None
        self.val_data = None

    def prepare_data(self, validation_weeks=8):
        """Prepare data for time series modeling with train/validation split"""
        print("=== PREPARING DATA FOR MODELING ===")

        # Clean and sort data
        self.data_clean = self.data.dropna(subset=["Weekly_Sales"])
        self.data_clean = self.data_clean.sort_values(["Store", "Dept", "Date"])

        # Time-based split
        unique_dates = sorted(self.data_clean["Date"].unique())
        split_date = unique_dates[-validation_weeks]

        self.train_data = self.data_clean[self.data_clean["Date"] < split_date].copy()
        self.val_data = self.data_clean[self.data_clean["Date"] >= split_date].copy()

        # Select features (excluding target and identifier columns)
        exclude_cols = ["Weekly_Sales", "Store", "Dept", "Date"]
        self.feature_columns = [
            col
            for col in self.data_clean.columns
            if col not in exclude_cols and not col.endswith("_scaled")
        ]

        print(f"Data split completed:")
        print(f"  - Training data: {self.train_data.shape}")
        print(f"  - Validation data: {self.val_data.shape}")
        print(f"  - Feature columns: {self.feature_columns}")
        print(
            f"  - Date range: {self.data_clean['Date'].min()} to {self.data_clean['Date'].max()}"
        )
        print(f"  - Split date: {split_date}")

        return self.train_data, self.val_data

    def temporal_fusion_transformer_advanced(self, sequence_length=5, epochs=50):
        """Advanced Temporal Fusion Transformer with full architecture"""
        print("=== TRAINING ADVANCED TEMPORAL FUSION TRANSFORMER ===")
        start_time = time.time()

        try:
            # Variable Selection Network
            class VariableSelectionNetwork(tf.keras.layers.Layer):
                def __init__(self, num_features, hidden_size, dropout_rate=0.1):
                    super(VariableSelectionNetwork, self).__init__()
                    self.num_features = num_features
                    self.hidden_size = hidden_size

                    self.linear1 = Dense(hidden_size, activation="relu")
                    self.linear2 = Dense(num_features, activation="softmax")
                    self.dropout = Dropout(dropout_rate)

                def call(self, inputs, training=None):
                    # inputs: [batch_size, time_steps, num_features]
                    batch_size = tf.shape(inputs)[0]
                    time_steps = tf.shape(inputs)[1]

                    # Flatten time dimension for processing
                    flattened = tf.reshape(inputs, [-1, self.num_features])

                    # Variable selection
                    x = self.linear1(flattened)
                    x = self.dropout(x, training=training)
                    weights = self.linear2(x)

                    # Reshape back
                    weights = tf.reshape(
                        weights, [batch_size, time_steps, self.num_features]
                    )

                    # Apply variable selection
                    selected = inputs * weights
                    return selected, weights

            # Gated Residual Network
            class GatedResidualNetwork(tf.keras.layers.Layer):
                def __init__(self, hidden_size, dropout_rate=0.1):
                    super(GatedResidualNetwork, self).__init__()
                    self.hidden_size = hidden_size

                    self.linear1 = Dense(hidden_size, activation="relu")
                    self.linear2 = Dense(hidden_size)
                    self.gate = Dense(hidden_size, activation="sigmoid")
                    self.dropout = Dropout(dropout_rate)
                    self.layer_norm = LayerNormalization()

                def call(self, inputs, training=None):
                    x = self.linear1(inputs)
                    x = self.dropout(x, training=training)
                    x = self.linear2(x)

                    gate = self.gate(inputs)

                    # Gated residual connection
                    output = gate * x + (1 - gate) * inputs
                    output = self.layer_norm(output)

                    return output

            # Prepare data for TFT
            agg_train = (
                self.train_data.groupby(["Store", "Date"])
                .agg(
                    {
                        "Weekly_Sales": "sum",
                        "Temperature": "mean",
                        "Fuel_Price": "mean",
                        "CPI": "mean",
                        "Unemployment": "mean",
                        "IsHoliday": "max",
                        "Total_MarkDown": "sum",
                        "Holiday_Weight": "max",
                        "Size": "first",
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
                        "CPI": "mean",
                        "Unemployment": "mean",
                        "IsHoliday": "max",
                        "Total_MarkDown": "sum",
                        "Holiday_Weight": "max",
                        "Size": "first",
                    }
                )
                .reset_index()
            )

            # Features
            continuous_features = [
                "Temperature",
                "Fuel_Price",
                "CPI",
                "Unemployment",
                "Total_MarkDown",
                "Size",
            ]
            categorical_features = ["IsHoliday"]

            # Create sequences
            def create_tft_sequences(data, seq_len):
                X_cont, X_cat, y, weights = [], [], [], []

                for store in data["Store"].unique():
                    store_data = data[data["Store"] == store].sort_values("Date")

                    if len(store_data) < seq_len + 1:
                        continue

                    for i in range(seq_len, len(store_data)):
                        X_cont.append(
                            store_data[continuous_features].iloc[i - seq_len : i].values
                        )
                        X_cat.append(
                            store_data[categorical_features]
                            .iloc[i - seq_len : i]
                            .values
                        )
                        y.append(store_data["Weekly_Sales"].iloc[i])
                        weights.append(store_data["Holiday_Weight"].iloc[i])

                return np.array(X_cont), np.array(X_cat), np.array(y), np.array(weights)

            X_cont_train, X_cat_train, y_train, train_weights = create_tft_sequences(
                agg_train, sequence_length
            )
            X_cont_val, X_cat_val, y_val, val_weights = create_tft_sequences(
                agg_val, sequence_length
            )

            if len(X_cont_train) == 0 or len(X_cont_val) == 0:
                print("Insufficient data for TFT")
                return None, None

            # Scale continuous features
            scaler_cont = StandardScaler()
            scaler_y = StandardScaler()

            X_cont_train_scaled = scaler_cont.fit_transform(
                X_cont_train.reshape(-1, X_cont_train.shape[-1])
            ).reshape(X_cont_train.shape)

            X_cont_val_scaled = scaler_cont.transform(
                X_cont_val.reshape(-1, X_cont_val.shape[-1])
            ).reshape(X_cont_val.shape)

            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

            # Build TFT model
            hidden_size = 64
            num_cont_features = len(continuous_features)
            num_cat_features = len(categorical_features)

            # Inputs
            cont_inputs = Input(
                shape=(sequence_length, num_cont_features), name="continuous"
            )
            cat_inputs = Input(
                shape=(sequence_length, num_cat_features), name="categorical"
            )

            # Embed categorical features
            cat_embedded = Dense(8, activation="relu")(cat_inputs)

            # Combine continuous and categorical
            combined = Concatenate(axis=-1)([cont_inputs, cat_embedded])

            # Variable Selection Network
            vsn = VariableSelectionNetwork(num_cont_features + 8, hidden_size)
            selected_features, variable_weights = vsn(combined)

            # Encoder-Decoder with attention
            # Encoder
            encoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True)
            encoder_outputs, state_h, state_c = encoder_lstm(selected_features)

            # Self-attention
            attention = MultiHeadAttention(num_heads=4, key_dim=hidden_size // 4)
            attention_output = attention(encoder_outputs, encoder_outputs)

            # Gated Residual Network
            grn = GatedResidualNetwork(hidden_size)
            grn_output = grn(attention_output)

            # Global pooling and output
            pooled = GlobalAveragePooling1D()(grn_output)

            # Final prediction layers
            dense1 = Dense(hidden_size, activation="relu")(pooled)
            dropout1 = Dropout(0.3)(dense1)
            dense2 = Dense(hidden_size // 2, activation="relu")(dropout1)
            dropout2 = Dropout(0.2)(dense2)
            output = Dense(1)(dropout2)

            # Create model
            model = Model(inputs=[cont_inputs, cat_inputs], outputs=output)

            model.compile(
                optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"]
            )

            # Train model
            callbacks = [
                EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
                ),
            ]

            history = model.fit(
                [X_cont_train_scaled, X_cat_train],
                y_train_scaled,
                validation_data=(
                    [X_cont_val_scaled, X_cat_val],
                    scaler_y.transform(y_val.reshape(-1, 1)).flatten(),
                ),
                epochs=epochs,
                batch_size=32,
                callbacks=callbacks,
                verbose=0,
            )

            # Make predictions
            y_pred_scaled = model.predict([X_cont_val_scaled, X_cat_val], verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

            training_time = time.time() - start_time

            self.models["TFT_Advanced"] = model
            self.results["TFT_Advanced"] = {
                "predictions": y_pred,
                "actual": y_val,
                "weights": val_weights,
                "training_time": training_time,
                "model_type": "Neural Network",
                "history": history,
                "variable_weights": None,  # Would extract from model
                "attention_weights": None,  # Would extract from model
            }

            print(f"Advanced TFT model trained in {training_time:.2f} seconds")
            return model, y_pred

        except Exception as e:
            print(f"Advanced TFT training failed: {e}")
            return None, None

    def ensemble_deep_learning_model(self, sequence_length=5, epochs=40):
        """Ensemble of multiple deep learning architectures - FIXED VERSION"""
        print("=== TRAINING ENSEMBLE DEEP LEARNING MODEL ===")
        start_time = time.time()

        try:
            # Prepare sequences
            features = [
                "Temperature",
                "Fuel_Price",
                "CPI",
                "Unemployment",
                "IsHoliday",
                "Total_MarkDown",
            ]

            def create_sequences(data, seq_len, features):
                X, y, weights = [], [], []

                # Aggregate by store and date for efficiency
                agg_data = (
                    data.groupby(["Store", "Date"])
                    .agg(
                        {
                            "Weekly_Sales": "sum",
                            **{
                                feat: "mean" if feat != "IsHoliday" else "max"
                                for feat in features
                            },
                            "Holiday_Weight": "max",
                        }
                    )
                    .reset_index()
                )

                for store in agg_data["Store"].unique():
                    store_data = agg_data[agg_data["Store"] == store].sort_values(
                        "Date"
                    )

                    if len(store_data) < seq_len + 1:
                        continue

                    for i in range(seq_len, len(store_data)):
                        X.append(store_data[features].iloc[i - seq_len : i].values)
                        y.append(store_data["Weekly_Sales"].iloc[i])
                        weights.append(store_data["Holiday_Weight"].iloc[i])

                return np.array(X), np.array(y), np.array(weights)

            X_train, y_train, train_weights = create_sequences(
                self.train_data, sequence_length, features
            )
            X_val, y_val, val_weights = create_sequences(
                self.val_data, sequence_length, features
            )

            if len(X_train) == 0 or len(X_val) == 0:
                print("Insufficient data for ensemble model")
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

            # Define multiple architectures
            def create_lstm_branch(inputs):
                x = LSTM(64, return_sequences=True)(inputs)
                x = Dropout(0.3)(x)
                x = LSTM(32, return_sequences=False)(x)
                x = Dropout(0.2)(x)
                return Dense(16, activation="relu")(x)

            def create_gru_branch(inputs):
                x = GRU(64, return_sequences=True)(inputs)
                x = Dropout(0.3)(x)
                x = GRU(32, return_sequences=False)(x)
                x = Dropout(0.2)(x)
                return Dense(16, activation="relu")(x)

            def create_cnn_branch(inputs):
                # FIXED: Adjusted for short sequences
                if sequence_length <= 3:
                    # For very short sequences, use single conv layer without pooling
                    x = Conv1D(
                        filters=32, kernel_size=2, activation="relu", padding="same"
                    )(inputs)
                    x = GlobalAveragePooling1D()(x)
                    return Dense(16, activation="relu")(x)
                else:
                    # For longer sequences, use the original approach but with padding
                    x = Conv1D(
                        filters=64, kernel_size=3, activation="relu", padding="same"
                    )(inputs)

                    # Only apply pooling if sequence length allows it
                    if sequence_length > 4:
                        x = MaxPooling1D(pool_size=2)(x)

                    # Adjust kernel size for second conv layer based on remaining sequence length
                    remaining_length = (
                        sequence_length // 2 if sequence_length > 4 else sequence_length
                    )
                    kernel_size = min(3, remaining_length)

                    if kernel_size >= 2:
                        x = Conv1D(
                            filters=32,
                            kernel_size=kernel_size,
                            activation="relu",
                            padding="same",
                        )(x)

                    x = GlobalAveragePooling1D()(x)
                    return Dense(16, activation="relu")(x)

            def create_attention_branch(inputs):
                # Ensure key_dim is reasonable for the sequence length
                key_dim = min(32, max(8, len(features) // 2))
                x = MultiHeadAttention(num_heads=4, key_dim=key_dim)(inputs, inputs)
                x = LayerNormalization()(x)
                x = GlobalAveragePooling1D()(x)
                return Dense(16, activation="relu")(x)

            # Input layer
            inputs = Input(shape=(sequence_length, len(features)))

            # Create branches
            lstm_branch = create_lstm_branch(inputs)
            gru_branch = create_gru_branch(inputs)
            cnn_branch = create_cnn_branch(inputs)
            attention_branch = create_attention_branch(inputs)

            # Combine branches
            combined = Concatenate()(
                [lstm_branch, gru_branch, cnn_branch, attention_branch]
            )

            # Final layers
            x = Dense(64, activation="relu")(combined)
            x = BatchNormalization()(x)
            x = Dropout(0.4)(x)
            x = Dense(32, activation="relu")(x)
            x = Dropout(0.3)(x)
            outputs = Dense(1)(x)

            # Create and compile model
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"]
            )

            print(f"Model input shape: {inputs.shape}")
            print(f"Sequence length: {sequence_length}, Features: {len(features)}")

            # Train model
            callbacks = [
                EarlyStopping(
                    monitor="val_loss", patience=15, restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6
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

            self.models["EnsembleDeep"] = model
            self.results["EnsembleDeep"] = {
                "predictions": y_pred,
                "actual": y_val,
                "weights": val_weights,
                "training_time": training_time,
                "model_type": "Neural Network",
                "history": history,
                "architecture": "LSTM+GRU+CNN+Attention",
            }

            print(
                f"Ensemble Deep Learning model trained in {training_time:.2f} seconds"
            )
            return model, y_pred

        except Exception as e:
            print(f"Ensemble Deep Learning training failed: {e}")
            return None, None

    def neural_ode_model(self, sequence_length=5, epochs=30):
        """Neural ODE model - CORRECTED VERSION to match TFT and Ensemble DL structure"""
        print("=== TRAINING NEURAL ODE MODEL (CORRECTED) ===")
        start_time = time.time()

        try:
            # Use same feature set as other models for consistency
            features = [
                "Temperature",
                "Fuel_Price",
                "CPI",
                "Unemployment",
                "IsHoliday",
                "Total_MarkDown",  # Added this feature like other models
            ]

            # Use the SAME sequence creation logic as ensemble model
            def create_sequences(data, seq_len, features):
                X, y, weights = [], [], []

                # Aggregate by store and date for efficiency (SAME AS ENSEMBLE)
                agg_data = (
                    data.groupby(["Store", "Date"])
                    .agg(
                        {
                            "Weekly_Sales": "sum",
                            **{
                                feat: "mean" if feat != "IsHoliday" else "max"
                                for feat in features
                            },
                            "Holiday_Weight": "max",
                        }
                    )
                    .reset_index()
                )

                for store in agg_data["Store"].unique():
                    store_data = agg_data[agg_data["Store"] == store].sort_values(
                        "Date"
                    )

                    if len(store_data) < seq_len + 1:
                        continue

                    for i in range(seq_len, len(store_data)):
                        X.append(store_data[features].iloc[i - seq_len : i].values)
                        y.append(store_data["Weekly_Sales"].iloc[i])
                        weights.append(store_data["Holiday_Weight"].iloc[i])

                return np.array(X), np.array(y), np.array(weights)

            # Create training and validation sequences (SAME AS OTHER MODELS)
            X_train, y_train, train_weights = create_sequences(
                self.train_data, sequence_length, features
            )
            X_val, y_val, val_weights = create_sequences(
                self.val_data, sequence_length, features
            )

            print(f"Training sequences: {len(X_train)}")
            print(f"Validation sequences: {len(X_val)}")

            if len(X_train) == 0 or len(X_val) == 0:
                print("Insufficient data for Neural ODE")
                return None, None

            # Scale data (SAME AS OTHER MODELS)
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_train_scaled = scaler_X.fit_transform(
                X_train.reshape(-1, X_train.shape[-1])
            ).reshape(X_train.shape)
            X_val_scaled = scaler_X.transform(
                X_val.reshape(-1, X_val.shape[-1])
            ).reshape(X_val.shape)
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

            # Neural ODE Architecture - IMPROVED
            inputs = Input(shape=(sequence_length, len(features)))

            # Initial processing layer
            hidden_dim = 64

            # Process temporal sequences first
            x = LSTM(hidden_dim, return_sequences=True, return_state=False)(inputs)
            x = LayerNormalization()(x)

            # Neural ODE blocks - simulating continuous dynamics
            def ode_residual_block(x, step_size=0.1):
                """
                Simulates one step of ODE integration using residual connections
                dx/dt = f(x, t) approximated as x_{t+1} = x_t + step_size * f(x_t, t)
                """
                residual = x  # x_t

                # f(x_t, t) - the derivative function
                dx = Dense(
                    hidden_dim,
                    activation="tanh",
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                )(x)
                dx = Dropout(0.1)(dx)  # Regularization
                dx = Dense(
                    hidden_dim,
                    activation="tanh",
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                )(dx)

                # Euler integration: x_{t+1} = x_t + step_size * dx/dt
                # We'll use step_size = 1 for simplicity, but could be learnable
                x_new = Add()([residual, dx])

                # Normalize to prevent exploding gradients
                return LayerNormalization()(x_new)

            # Apply multiple ODE integration steps
            num_ode_steps = 6  # Simulate 6 time steps of continuous dynamics
            for step in range(num_ode_steps):
                x = ode_residual_block(x, step_size=0.1)

            # Additional processing for better temporal modeling
            x = GRU(32, return_sequences=False)(x)  # Final temporal aggregation
            x = Dropout(0.3)(x)

            # Output layers
            x = Dense(64, activation="relu")(x)
            x = BatchNormalization()(x)
            x = Dropout(0.4)(x)
            x = Dense(32, activation="relu")(x)
            x = Dropout(0.3)(x)
            outputs = Dense(1)(x)

            # Create and compile model
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"]
            )

            print(f"Model input shape: {inputs.shape}")
            print(f"Sequence length: {sequence_length}, Features: {len(features)}")

            # Train model with SAME callbacks as other models
            callbacks = [
                EarlyStopping(
                    monitor="val_loss", patience=15, restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6
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
                batch_size=64,  # Same batch size as ensemble model
                callbacks=callbacks,
                verbose=0,
            )

            # Make predictions
            y_pred_scaled = model.predict(X_val_scaled, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

            training_time = time.time() - start_time

            # Store results in SAME format as other models
            self.models["NeuralODE"] = model
            self.results["NeuralODE"] = {
                "predictions": y_pred,
                "actual": y_val,
                "weights": val_weights,
                "training_time": training_time,
                "model_type": "Neural Network",
                "history": history,
                "architecture": "LSTM+ODE_Blocks+GRU",  # Added architecture info
            }

            print(f"Neural ODE model trained in {training_time:.2f} seconds")
            print(f"Predictions shape: {y_pred.shape}, Actual shape: {y_val.shape}")
            return model, y_pred

        except Exception as e:
            print(f"Neural ODE training failed: {e}")
            import traceback

            print(f"Detailed error: {traceback.format_exc()}")
            return None, None

    def state_space_model(self):
        """State Space Model using SARIMAX - COMPLETE FIXED VERSION"""
        print("=== TRAINING STATE SPACE MODEL (SARIMAX) - COMPLETE FIXED ===")
        start_time = time.time()

        try:
            # Prepare time series data with proper data type handling
            ts_data = (
                self.train_data.groupby("Date")
                .agg(
                    {
                        "Weekly_Sales": "sum",
                        "Temperature": "mean",
                        "Fuel_Price": "mean",
                        "CPI": "mean",
                        "Unemployment": "mean",
                        "IsHoliday": "max",
                    }
                )
                .reset_index()
            )

            # Debug: Check for data issues
            print(f"Training data shape before processing: {ts_data.shape}")
            print(f"Date range: {ts_data['Date'].min()} to {ts_data['Date'].max()}")

            # Check for missing values
            missing_counts = ts_data.isnull().sum()
            if missing_counts.any():
                print("Missing values found:")
                print(missing_counts[missing_counts > 0])

            # Ensure Date is datetime
            ts_data["Date"] = pd.to_datetime(ts_data["Date"])
            ts_data.set_index("Date", inplace=True)
            ts_data = ts_data.sort_index()

            # *** THIS IS THE KEY FIX: USE RESAMPLE INSTEAD OF ASFREQ ***
            print("Using resample instead of asfreq...")
            ts_data = ts_data.resample("W").agg(
                {
                    "Weekly_Sales": "sum",
                    "Temperature": "mean",
                    "Fuel_Price": "mean",
                    "CPI": "mean",
                    "Unemployment": "mean",
                    "IsHoliday": "max",
                }
            )

            print(f"Shape after resample: {ts_data.shape}")
            print(f"NaN values after resample: {ts_data.isnull().sum().sum()}")

            # Handle missing values BEFORE data type conversion
            if ts_data.isnull().any().any():
                print("Filling missing values...")
                ts_data = ts_data.fillna(method="ffill").fillna(method="bfill")

                # If still NaN, fill with column medians
                for col in ts_data.columns:
                    if ts_data[col].isnull().any():
                        median_val = ts_data[col].median()
                        ts_data[col] = ts_data[col].fillna(median_val)
                        print(f"Filled {col} with median: {median_val}")

            print(f"Shape after filling: {ts_data.shape}")

            # SAFE data type conversion - DON'T USE errors='coerce'
            exog_vars = [
                "Temperature",
                "Fuel_Price",
                "CPI",
                "Unemployment",
                "IsHoliday",
            ]

            print("Converting data types safely...")
            for col in ["Weekly_Sales"] + exog_vars:
                print(f"Converting {col}: {ts_data[col].dtype} -> float64")

                if col == "IsHoliday":
                    # Handle boolean column specially
                    if ts_data[col].dtype == "bool":
                        ts_data[col] = ts_data[col].astype(int).astype("float64")
                    elif ts_data[col].dtype == "object":
                        # Convert True/False strings to 1/0
                        ts_data[col] = ts_data[col].map(
                            {"True": 1, "False": 0, True: 1, False: 0}
                        )
                        ts_data[col] = ts_data[col].astype("float64")
                    else:
                        ts_data[col] = ts_data[col].astype("float64")
                else:
                    # For numeric columns, ensure they're float
                    if pd.api.types.is_numeric_dtype(ts_data[col]):
                        ts_data[col] = ts_data[col].astype("float64")
                    else:
                        # Only use coerce as last resort and fill immediately
                        original_count = len(ts_data)
                        ts_data[col] = pd.to_numeric(ts_data[col], errors="coerce")
                        nan_count = ts_data[col].isnull().sum()

                        if nan_count > 0:
                            print(
                                f"  WARNING: {nan_count} values converted to NaN in {col}"
                            )
                            median_val = ts_data[col].median()
                            ts_data[col] = ts_data[col].fillna(median_val)
                            print(f"  Filled with median: {median_val}")

                        ts_data[col] = ts_data[col].astype("float64")

            # Final check - should have NO NaNs and same shape
            print(f"Final training shape: {ts_data.shape}")
            print(f"Final training NaN count: {ts_data.isnull().sum().sum()}")
            print(f"Training data types: {ts_data.dtypes}")

            if ts_data.empty:
                print("ERROR: Training data is empty!")
                return None, None

            if ts_data.isnull().any().any():
                print("ERROR: Still have NaN values in training data!")
                print(ts_data.isnull().sum())
                return None, None

            # Verify we have enough data
            if len(ts_data) < 10:
                print("ERROR: Insufficient training data for SARIMAX model")
                return None, None

            # Fit SARIMAX model with error handling
            try:
                model = SARIMAX(
                    ts_data["Weekly_Sales"].values,  # Convert to numpy array explicitly
                    exog=ts_data[exog_vars].values,  # Convert to numpy array explicitly
                    order=(1, 1, 1),  # ARIMA order
                    seasonal_order=(1, 1, 1, 52),  # Seasonal order (weekly seasonality)
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )

                print("Fitting SARIMAX model...")
                fitted_model = model.fit(disp=False, maxiter=100)

            except Exception as model_error:
                print(
                    f"SARIMAX fitting failed with seasonal order, trying simpler model: {model_error}"
                )
                # Try simpler model without seasonal component
                model = SARIMAX(
                    ts_data["Weekly_Sales"].values,
                    exog=ts_data[exog_vars].values,
                    order=(1, 1, 1),  # ARIMA order only
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted_model = model.fit(disp=False, maxiter=100)

            # Prepare validation data with same processing
            val_ts_data = (
                self.val_data.groupby("Date")
                .agg(
                    {
                        "Weekly_Sales": "sum",
                        "Temperature": "mean",
                        "Fuel_Price": "mean",
                        "CPI": "mean",
                        "Unemployment": "mean",
                        "IsHoliday": "max",
                    }
                )
                .reset_index()
            )

            print(f"Validation data shape before processing: {val_ts_data.shape}")

            # Process validation data - USE SAME SAFE APPROACH AS TRAINING
            val_ts_data["Date"] = pd.to_datetime(val_ts_data["Date"])
            val_ts_data.set_index("Date", inplace=True)
            val_ts_data = val_ts_data.sort_index()

            # *** USE RESAMPLE FOR VALIDATION TOO ***
            print("Using resample for validation data...")
            val_ts_data = val_ts_data.resample("W").agg(
                {
                    "Weekly_Sales": "sum",
                    "Temperature": "mean",
                    "Fuel_Price": "mean",
                    "CPI": "mean",
                    "Unemployment": "mean",
                    "IsHoliday": "max",
                }
            )

            print(f"Validation shape after resample: {val_ts_data.shape}")
            print(
                f"Validation NaN values after resample: {val_ts_data.isnull().sum().sum()}"
            )

            # Handle missing values in validation data BEFORE type conversion
            if val_ts_data.isnull().any().any():
                print("Filling validation missing values...")
                val_ts_data = val_ts_data.fillna(method="ffill").fillna(method="bfill")

                # If still NaN, fill with column medians
                for col in val_ts_data.columns:
                    if val_ts_data[col].isnull().any():
                        median_val = val_ts_data[col].median()
                        val_ts_data[col] = val_ts_data[col].fillna(median_val)
                        print(f"Filled validation {col} with median: {median_val}")

            # SAFE data type conversion for validation - DON'T USE errors='coerce'
            print("Converting validation data types safely...")
            for col in ["Weekly_Sales"] + exog_vars:
                print(
                    f"Converting validation {col}: {val_ts_data[col].dtype} -> float64"
                )

                if col == "IsHoliday":
                    # Handle boolean column specially
                    if val_ts_data[col].dtype == "bool":
                        val_ts_data[col] = (
                            val_ts_data[col].astype(int).astype("float64")
                        )
                    elif val_ts_data[col].dtype == "object":
                        # Convert True/False strings to 1/0
                        val_ts_data[col] = val_ts_data[col].map(
                            {"True": 1, "False": 0, True: 1, False: 0}
                        )
                        val_ts_data[col] = val_ts_data[col].astype("float64")
                    else:
                        val_ts_data[col] = val_ts_data[col].astype("float64")
                else:
                    # For numeric columns, ensure they're float
                    if pd.api.types.is_numeric_dtype(val_ts_data[col]):
                        val_ts_data[col] = val_ts_data[col].astype("float64")
                    else:
                        # Only use coerce as last resort and fill immediately
                        val_ts_data[col] = pd.to_numeric(
                            val_ts_data[col], errors="coerce"
                        )
                        nan_count = val_ts_data[col].isnull().sum()

                        if nan_count > 0:
                            print(
                                f"  WARNING: {nan_count} validation values converted to NaN in {col}"
                            )
                            median_val = val_ts_data[col].median()
                            val_ts_data[col] = val_ts_data[col].fillna(median_val)
                            print(f"  Filled with median: {median_val}")

                        val_ts_data[col] = val_ts_data[col].astype("float64")

            # Final validation check
            print(f"Final validation shape: {val_ts_data.shape}")
            print(f"Final validation NaN count: {val_ts_data.isnull().sum().sum()}")

            if val_ts_data.empty:
                print("ERROR: Validation data is empty!")
                return None, None

            if val_ts_data.isnull().any().any():
                print("ERROR: Still have NaN values in validation data!")
                print(val_ts_data.isnull().sum())
                return None, None

            # Make predictions
            forecast_steps = len(val_ts_data)
            if forecast_steps == 0:
                print("ERROR: No validation data available")
                return None, None

            print(f"Forecasting {forecast_steps} steps...")

            # Ensure exogenous variables are numeric arrays
            exog_forecast = val_ts_data[exog_vars].values.astype("float64")

            # Check for any issues with exogenous data
            if np.isnan(exog_forecast).any():
                print("WARNING: NaN values in exogenous forecast data")
                return None, None

            forecast = fitted_model.forecast(steps=forecast_steps, exog=exog_forecast)

            training_time = time.time() - start_time

            # Store results
            self.models["SARIMAX"] = fitted_model
            self.results["SARIMAX"] = {
                "predictions": (
                    forecast.values if hasattr(forecast, "values") else forecast
                ),
                "actual": val_ts_data["Weekly_Sales"].values,
                "weights": np.ones(len(forecast)),
                "training_time": training_time,
                "model_type": "Statistical",
                "model_summary": str(
                    fitted_model.summary()
                ),  # Convert to string to avoid issues
            }

            print(f"SARIMAX model trained successfully in {training_time:.2f} seconds")
            print(f"Forecast shape: {forecast.shape}")
            print(f"Actual shape: {val_ts_data['Weekly_Sales'].values.shape}")

            return fitted_model, (
                forecast.values if hasattr(forecast, "values") else forecast
            )

        except Exception as e:
            print(f"SARIMAX training failed: {e}")
            import traceback

            print(f"Detailed error: {traceback.format_exc()}")
            return None, None

    def gaussian_process_model(self):
        """Gaussian Process for time series forecasting (simplified using sklearn)"""
        print("=== TRAINING GAUSSIAN PROCESS MODEL ===")
        start_time = time.time()

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

            # Prepare data
            features = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "IsHoliday"]

            # Aggregate by date
            train_agg = (
                self.train_data.groupby("Date")
                .agg(
                    {
                        "Weekly_Sales": "sum",
                        **{
                            feat: "mean" if feat != "IsHoliday" else "max"
                            for feat in features
                        },
                    }
                )
                .reset_index()
            )

            val_agg = (
                self.val_data.groupby("Date")
                .agg(
                    {
                        "Weekly_Sales": "sum",
                        **{
                            feat: "mean" if feat != "IsHoliday" else "max"
                            for feat in features
                        },
                    }
                )
                .reset_index()
            )

            # Prepare features and target
            X_train = train_agg[features].values
            y_train = train_agg["Weekly_Sales"].values
            X_val = val_agg[features].values
            y_val = val_agg["Weekly_Sales"].values

            # Scale features
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_train_scaled = scaler_X.fit_transform(X_train)
            X_val_scaled = scaler_X.transform(X_val)
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

            # Define kernel
            kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)

            # Create and fit GP model
            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=False,
                n_restarts_optimizer=3,
                random_state=42,
            )

            # Fit model (subsample for computational efficiency)
            if len(X_train_scaled) > 200:
                indices = np.random.choice(len(X_train_scaled), 200, replace=False)
                X_train_sub = X_train_scaled[indices]
                y_train_sub = y_train_scaled[indices]
            else:
                X_train_sub = X_train_scaled
                y_train_sub = y_train_scaled

            model.fit(X_train_sub, y_train_sub)

            # Make predictions with uncertainty
            y_pred_scaled, y_std_scaled = model.predict(X_val_scaled, return_std=True)

            # Denormalize
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_std = y_std_scaled * scaler_y.scale_[0]

            training_time = time.time() - start_time

            self.models["GaussianProcess"] = model
            self.results["GaussianProcess"] = {
                "predictions": y_pred,
                "actual": y_val,
                "uncertainty": y_std,
                "weights": np.ones(len(y_pred)),
                "training_time": training_time,
                "model_type": "Probabilistic",
            }

            print(f"Gaussian Process model trained in {training_time:.2f} seconds")
            return model, y_pred

        except Exception as e:
            print(f"Gaussian Process training failed: {e}")
            return None, None


# Usage example and test functions
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

    # Assuming you have data from WalmartDataLoader
    eda = WalmartComprehensiveEDA(train_data, test_data, features_data, stores_data)
    merged_data = eda.merge_datasets()

    feature_eng = WalmartFeatureEngineering(merged_data)
    processed_data = feature_eng.create_walmart_features()
    processed_data = feature_eng.handle_missing_values()
    print("Feature Engineering class ready!")

    advanced_models = AdvancedWalmartForecastingModels(processed_data)
    train_data, val_data = advanced_models.prepare_data()

    # Train various models
    tft_model, tft_pred = advanced_models.temporal_fusion_transformer_advanced()
    ensemble_model, ensemble_pred = advanced_models.ensemble_deep_learning_model()
    ode_model, ode_pred = advanced_models.neural_ode_model()
    sarimax_model, sarimax_pred = advanced_models.state_space_model()
    gp_model, gp_pred = advanced_models.gaussian_process_model()
