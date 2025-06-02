import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class WalmartFeatureEngineering:
    """Advanced feature engineering specific to Walmart competition"""

    def __init__(self, merged_data):
        self.data = merged_data.copy()
        self.feature_importance = {}

    def create_walmart_features(self):
        """Create features specific to Walmart competition"""
        print("=== WALMART-SPECIFIC FEATURE ENGINEERING ===")

        # Sort data for time-based features
        self.data = self.data.sort_values(["Store", "Dept", "Date"])

        # 1. Holiday-related features
        self.data["Holiday_Weight"] = self.data["IsHoliday"].apply(
            lambda x: 5.0 if x == 1 else 1.0
        )

        # Pre/post holiday indicators
        self.data["Pre_Holiday"] = (
            self.data.groupby(["Store", "Dept"])["IsHoliday"].shift(-1).fillna(0)
        )
        self.data["Post_Holiday"] = (
            self.data.groupby(["Store", "Dept"])["IsHoliday"].shift(1).fillna(0)
        )

        # 2. Temporal features
        self.data["Year"] = self.data["Date"].dt.year
        self.data["Month"] = self.data["Date"].dt.month
        self.data["Week"] = self.data["Date"].dt.isocalendar().week
        self.data["Quarter"] = self.data["Date"].dt.quarter

        # Cyclical encoding
        self.data["Month_sin"] = np.sin(2 * np.pi * self.data["Month"] / 12)
        self.data["Month_cos"] = np.cos(2 * np.pi * self.data["Month"] / 12)
        self.data["Week_sin"] = np.sin(2 * np.pi * self.data["Week"] / 52)
        self.data["Week_cos"] = np.cos(2 * np.pi * self.data["Week"] / 52)

        # 3. Lag features (critical for time series)
        lag_periods = [1, 2, 4, 8, 12, 52]  # Including yearly lag
        for lag in lag_periods:
            self.data[f"Sales_lag_{lag}"] = self.data.groupby(["Store", "Dept"])[
                "Weekly_Sales"
            ].shift(lag)

        # 4. Rolling statistics
        windows = [4, 8, 12, 24, 52]
        for window in windows:
            self.data[f"Sales_rolling_mean_{window}"] = (
                self.data.groupby(["Store", "Dept"])["Weekly_Sales"]
                .rolling(window=window)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
            self.data[f"Sales_rolling_std_{window}"] = (
                self.data.groupby(["Store", "Dept"])["Weekly_Sales"]
                .rolling(window=window)
                .std()
                .reset_index(level=[0, 1], drop=True)
            )
            self.data[f"Sales_rolling_max_{window}"] = (
                self.data.groupby(["Store", "Dept"])["Weekly_Sales"]
                .rolling(window=window)
                .max()
                .reset_index(level=[0, 1], drop=True)
            )
            self.data[f"Sales_rolling_min_{window}"] = (
                self.data.groupby(["Store", "Dept"])["Weekly_Sales"]
                .rolling(window=window)
                .min()
                .reset_index(level=[0, 1], drop=True)
            )

        # 5. Markdown features
        markdown_cols = [
            "MarkDown1",
            "MarkDown2",
            "MarkDown3",
            "MarkDown4",
            "MarkDown5",
        ]

        # Fill markdown NaN with 0 (no promotion)
        for col in markdown_cols:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna(0)

        # Total markdown
        self.data["Total_MarkDown"] = self.data[markdown_cols].sum(axis=1)

        # Markdown indicators
        for col in markdown_cols:
            if col in self.data.columns:
                self.data[f"{col}_Active"] = (self.data[col] > 0).astype(int)

        # Markdown intensity
        self.data["MarkDown_Intensity"] = self.data["Total_MarkDown"] / (
            self.data["Size"] + 1
        )

        # 6. Store and department-level features
        # Store performance relative to store type average
        store_type_avg = self.data.groupby("Type")["Weekly_Sales"].mean().to_dict()
        self.data["Store_Type_Avg"] = self.data["Type"].map(store_type_avg)

        # Department performance relative to department average
        dept_avg = self.data.groupby("Dept")["Weekly_Sales"].mean().to_dict()
        self.data["Dept_Avg"] = self.data["Dept"].map(dept_avg)

        # Store-department interaction
        store_dept_avg = (
            self.data.groupby(["Store", "Dept"])["Weekly_Sales"].mean().to_dict()
        )
        self.data["Store_Dept_Avg"] = self.data.set_index(["Store", "Dept"]).index.map(
            store_dept_avg
        )

        # 7. Economic interaction features
        self.data["Unemployment_Temperature"] = (
            self.data["Unemployment"] * self.data["Temperature"]
        )
        self.data["CPI_Fuel_Interaction"] = self.data["CPI"] * self.data["Fuel_Price"]

        # Economic stress indicator
        self.data["Economic_Stress"] = (
            self.data["Unemployment"] - self.data["Unemployment"].mean()
        ) / self.data["Unemployment"].std() + (
            self.data["Fuel_Price"] - self.data["Fuel_Price"].mean()
        ) / self.data[
            "Fuel_Price"
        ].std()

        # 8. Trend features
        # Linear trend for each store-department combination
        def calculate_trend(group):
            if len(group) < 3:
                return pd.Series([0] * len(group), index=group.index)
            x = np.arange(len(group))
            slope = np.polyfit(x, group.values, 1)[0]
            return pd.Series([slope] * len(group), index=group.index)

        self.data["Sales_Trend"] = (
            self.data.groupby(["Store", "Dept"])["Weekly_Sales"]
            .apply(calculate_trend)
            .reset_index(level=[0, 1], drop=True)
        )

        print(f"Feature engineering completed. New shape: {self.data.shape}")
        print(
            f"Added {self.data.shape[1] - len(['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday'])} new features"
        )

        return self.data

    def handle_missing_values(self):
        """Handle missing values with Walmart-specific logic"""
        print("=== HANDLING MISSING VALUES ===")

        # Markdown columns: NaN means no promotion (fill with 0)
        markdown_cols = [
            "MarkDown1",
            "MarkDown2",
            "MarkDown3",
            "MarkDown4",
            "MarkDown5",
        ]
        for col in markdown_cols:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna(0)

        # For lag and rolling features, use forward fill within store-dept groups
        lag_cols = [
            col for col in self.data.columns if "lag_" in col or "rolling_" in col
        ]
        for col in lag_cols:
            self.data[col] = self.data.groupby(["Store", "Dept"])[col].fillna(
                method="ffill"
            )

        # For remaining missing values, use interpolation
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.data[col].isnull().sum() > 0:
                self.data[col] = (
                    self.data.groupby(["Store", "Dept"])[col]
                    .apply(lambda x: x.interpolate(method="linear"))
                    .reset_index(level=[0, 1], drop=True)
                )

        # Final cleanup: fill any remaining NaN
        self.data = self.data.fillna(method="ffill").fillna(method="bfill").fillna(0)

        print(
            f"Missing values handled. Remaining NaN: {self.data.isnull().sum().sum()}"
        )
        return self.data

    def scale_features(self, feature_columns):
        """Scale features for neural network models"""
        print("=== SCALING FEATURES ===")

        scalers = {}
        for col in feature_columns:
            if col in self.data.columns:
                scaler = StandardScaler()
                self.data[f"{col}_scaled"] = scaler.fit_transform(self.data[[col]])
                scalers[col] = scaler

        print(f"Scaled {len(feature_columns)} features")
        return self.data, scalers


if __name__ == "__main__":
    # Assuming you have merged_data from previous steps
    from src.data_loader import WalmartDataLoader
    from src.data_processing import WalmartComprehensiveEDA

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
