import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class WalmartDataLoader:
    """
    Data loader for actual Walmart sales dataset from Kaggle
    Dataset: Walmart Recruiting - Store Sales Forecasting
    """

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.features_data = None
        self.stores_data = None
        self.holiday_weights = None

    def load_data(self):
        """Load all CSV files"""
        try:
            self.train_data = pd.read_csv("data/train.csv")
            self.test_data = pd.read_csv("data/test.csv")
            self.stores_data = pd.read_csv("data/stores.csv")
            self.features_data = pd.read_csv("data/features.csv")

            # Convert dates
            self.train_data["Date"] = pd.to_datetime(self.train_data["Date"])
            self.test_data["Date"] = pd.to_datetime(self.test_data["Date"])
            self.features_data["Date"] = pd.to_datetime(self.features_data["Date"])

            print("All datasets loaded successfully!")
            return True
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            print(
                "Please make sure all CSV files (train.csv, test.csv, stores.csv, features.csv) are in the working directory."
            )
            return False

    def basic_info(self):
        """Display basic information about all datasets"""
        print("=" * 80)
        print("WALMART SALES FORECASTING - DATASET OVERVIEW")
        print("=" * 80)

        datasets = {
            "Training Data": self.train_data,
            "Test Data": self.test_data,
            "Stores Data": self.stores_data,
            "Features Data": self.features_data,
        }

        for name, df in datasets.items():
            if df is not None:
                print(f"\n {name}:")
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)}")
                print(
                    f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
                )

                if name == "Training Data":
                    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
                    print(f"   Unique stores: {df['Store'].nunique()}")
                    print(f"   Unique departments: {df['Dept'].nunique()}")
                    print(f"   Total sales records: {len(df):,}")


if __name__ == "__main__":
    loader = WalmartDataLoader()
    loader.load_data()
    loader.basic_info()

    print("Data loading complete!")
