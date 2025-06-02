import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")

# Set consistent style and color palette for all visualizations
plt.style.use("default")
sns.set_palette("Set2")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

# Define consistent color scheme
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent1": "#F18F01",
    "accent2": "#C73E1D",
    "neutral": "#6C757D",
    "success": "#198754",
    "warning": "#FFC107",
    "light": "#F8F9FA",
    "palette": [
        "#2E86AB",
        "#A23B72",
        "#F18F01",
        "#C73E1D",
        "#6C757D",
        "#198754",
        "#FFC107",
        "#17A2B8",
    ],
}


class WalmartComprehensiveEDA:
    """Enhanced EDA combining business insights with advanced time series analysis"""

    def __init__(
        self, train_data=None, test_data=None, features_data=None, stores_data=None
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.features_data = features_data
        self.stores_data = stores_data
        self.merged_data = None
        self.holiday_weeks = {
            "Super Bowl": ["2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"],
            "Labor Day": ["2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"],
            "Thanksgiving": ["2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"],
            "Christmas": ["2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"],
        }

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
            print("Please make sure all CSV files are in the data directory.")
            return False

    def merge_datasets(self):
        """Merge all datasets for comprehensive analysis"""
        print("=" * 60)
        print("MERGING DATASETS")
        print("=" * 60)

        # Merge training data with features
        self.merged_data = pd.merge(
            self.train_data,
            self.features_data,
            on=["Store", "Date"],
            how="left",
            suffixes=("", "_feat"),
        )

        # Use IsHoliday from training data (more reliable)
        if "IsHoliday_feat" in self.merged_data.columns:
            self.merged_data.drop("IsHoliday_feat", axis=1, inplace=True)

        # Merge with store information
        self.merged_data = pd.merge(
            self.merged_data, self.stores_data, on="Store", how="left"
        )

        print(f"Merged training data shape: {self.merged_data.shape}")
        print(
            f"Date range: {self.merged_data['Date'].min()} to {self.merged_data['Date'].max()}"
        )
        print(f"Number of stores: {self.merged_data['Store'].nunique()}")
        print(f"Number of departments: {self.merged_data['Dept'].nunique()}")

        return self.merged_data

    def basic_info_and_quality(self):
        """Enhanced basic information and data quality assessment"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE DATA OVERVIEW")
        print("=" * 60)

        # Basic dataset information
        datasets = {
            "Training Data": self.train_data,
            "Test Data": self.test_data,
            "Stores Data": self.stores_data,
            "Features Data": self.features_data,
            "Merged Data": self.merged_data,
        }

        for name, df in datasets.items():
            if df is not None:
                print(f"\n{name}:")
                print(f"   Shape: {df.shape}")
                print(
                    f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
                )

                if "Date" in df.columns:
                    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")

        # Data quality assessment
        print("\nDATA QUALITY ASSESSMENT:")

        if self.merged_data is not None:
            # Sales data quality
            total_records = len(self.merged_data)
            negative_sales = (self.merged_data["Weekly_Sales"] < 0).sum()
            zero_sales = (self.merged_data["Weekly_Sales"] == 0).sum()
            positive_sales = (self.merged_data["Weekly_Sales"] > 0).sum()

            print(f"   Total records: {total_records:,}")
            print(
                f"   Positive sales: {positive_sales:,} ({positive_sales/total_records*100:.1f}%)"
            )
            print(
                f"   Zero sales: {zero_sales:,} ({zero_sales/total_records*100:.1f}%)"
            )
            print(
                f"   Negative sales: {negative_sales:,} ({negative_sales/total_records*100:.1f}%)"
            )

            if negative_sales > 0:
                print(
                    f"   WARNING: Found {negative_sales:,} records with negative sales!"
                )

            # Sales statistics
            print("\nSALES STATISTICS:")
            sales_stats = self.merged_data["Weekly_Sales"].describe()
            print(sales_stats)

        return datasets

    def missing_values_comprehensive(self):
        """Enhanced missing values analysis"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE MISSING VALUES ANALYSIS")
        print("=" * 60)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()

        datasets = {
            "Training Data": self.train_data,
            "Features Data": self.features_data,
            "Stores Data": self.stores_data,
            "Merged Data": self.merged_data,
        }

        missing_summary = {}

        for i, (name, df) in enumerate(datasets.items()):
            if df is not None and i < 4:
                missing_data = df.isnull().sum()
                missing_percent = (missing_data / len(df)) * 100

                missing_summary[name] = {
                    "total_missing": missing_data.sum(),
                    "columns_with_missing": (missing_data > 0).sum(),
                    "worst_column": (
                        missing_data.idxmax() if missing_data.sum() > 0 else "None"
                    ),
                    "worst_percentage": missing_percent.max(),
                }

                print(f"\n{name}:")
                if missing_data.sum() == 0:
                    print("   No missing values!")
                else:
                    for col in missing_data[missing_data > 0].index:
                        print(
                            f"   • {col}: {missing_data[col]:,} ({missing_percent[col]:.2f}%)"
                        )

                # Visualization
                if missing_data.sum() > 0:
                    missing_percent[missing_percent > 0].plot(
                        kind="bar", ax=axes[i], color=COLORS["primary"], alpha=0.8
                    )
                    axes[i].set_title(f"{name} - Missing Values %", fontweight="bold")
                    axes[i].set_ylabel("Percentage Missing")
                    axes[i].tick_params(axis="x", rotation=45)
                    axes[i].grid(True, alpha=0.3)
                else:
                    axes[i].text(
                        0.5,
                        0.5,
                        "No Missing Values",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=axes[i].transAxes,
                        fontsize=14,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["light"]),
                    )
                    axes[i].set_title(f"{name} - Complete Data", fontweight="bold")

        plt.tight_layout()
        plt.show()

        return missing_summary

    def competition_specific_analysis(self):
        """Analysis specific to Walmart competition requirements"""
        print("\n" + "=" * 60)
        print("WALMART COMPETITION ANALYSIS")
        print("=" * 60)

        if self.merged_data is None:
            print("Please run merge_datasets() first")
            return

        # Holiday multiplier calculation
        holiday_sales = self.merged_data.groupby("IsHoliday")["Weekly_Sales"].agg(
            ["mean", "sum", "count", "std"]
        )
        print("\nHOLIDAY IMPACT ANALYSIS:")
        print(holiday_sales)

        holiday_multiplier = (
            holiday_sales.iloc[1]["mean"] / holiday_sales.iloc[0]["mean"]
        )
        print(f"\nHoliday sales multiplier: {holiday_multiplier:.3f}x")

        holiday_boost = (
            (holiday_sales.iloc[1]["mean"] - holiday_sales.iloc[0]["mean"])
            / holiday_sales.iloc[0]["mean"]
        ) * 100
        print(f"Holiday sales boost: +{holiday_boost:.1f}%")

        # Store type performance analysis
        print("\nSTORE TYPE PERFORMANCE:")
        store_type_perf = self.merged_data.groupby("Type")["Weekly_Sales"].agg(
            ["mean", "std", "count", "sum"]
        )
        print(store_type_perf)

        # Department performance analysis
        dept_performance = (
            self.merged_data.groupby("Dept")["Weekly_Sales"]
            .agg(["mean", "sum", "count", "std"])
            .round(2)
        )
        dept_performance = dept_performance.sort_values("sum", ascending=False)

        print(f"\nTOP 10 DEPARTMENTS BY TOTAL SALES:")
        print(dept_performance.head(10))

        print(f"\nBOTTOM 10 DEPARTMENTS BY TOTAL SALES:")
        print(dept_performance.tail(10))

        # Markdown effectiveness analysis
        markdown_cols = [
            "MarkDown1",
            "MarkDown2",
            "MarkDown3",
            "MarkDown4",
            "MarkDown5",
        ]
        print(f"\nMARKDOWN EFFECTIVENESS ANALYSIS:")

        markdown_analysis = {}
        for col in markdown_cols:
            if col in self.merged_data.columns:
                available_pct = (
                    self.merged_data[col].notna().sum() / len(self.merged_data)
                ) * 100
                avg_markdown = self.merged_data[col].mean()

                # Sales with vs without markdown
                with_markdown = self.merged_data[self.merged_data[col].notna()][
                    "Weekly_Sales"
                ].mean()
                without_markdown = self.merged_data[self.merged_data[col].isna()][
                    "Weekly_Sales"
                ].mean()

                markdown_analysis[col] = {
                    "availability_pct": available_pct,
                    "avg_value": avg_markdown,
                    "sales_with": with_markdown,
                    "sales_without": without_markdown,
                    "impact_ratio": (
                        with_markdown / without_markdown if without_markdown > 0 else 0
                    ),
                }

                print(
                    f"{col}: {available_pct:.1f}% available, avg: ${avg_markdown:,.0f}, "
                    f"impact ratio: {markdown_analysis[col]['impact_ratio']:.3f}"
                )

        return {
            "holiday_multiplier": holiday_multiplier,
            "holiday_boost": holiday_boost,
            "store_type_performance": store_type_perf,
            "dept_performance": dept_performance,
            "markdown_analysis": markdown_analysis,
        }

    def advanced_time_series_analysis(self):
        """Advanced time series analysis with decomposition and stationarity testing"""
        print("\n" + "=" * 60)
        print("ADVANCED TIME SERIES ANALYSIS")
        print("=" * 60)

        if self.merged_data is None:
            print("Please run merge_datasets() first")
            return

        # Aggregate sales by date for time series analysis
        ts_data = self.merged_data.groupby("Date")["Weekly_Sales"].sum().reset_index()
        ts_data.set_index("Date", inplace=True)
        ts_series = ts_data["Weekly_Sales"]

        print(f"Time series shape: {ts_series.shape}")
        print(f"Date range: {ts_series.index.min()} to {ts_series.index.max()}")
        print(f"Frequency: Weekly data points")

        # 1. Time Series Decomposition
        print("\n1. SEASONAL DECOMPOSITION:")
        try:
            decomposition = seasonal_decompose(ts_series, model="additive", period=52)

            # Plot decomposition
            fig, axes = plt.subplots(4, 1, figsize=(16, 14))

            decomposition.observed.plot(
                ax=axes[0],
                title="Original Time Series",
                color=COLORS["primary"],
                linewidth=2,
            )
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title("Original Time Series", fontweight="bold")

            decomposition.trend.plot(
                ax=axes[1],
                title="Trend Component",
                color=COLORS["success"],
                linewidth=2,
            )
            axes[1].grid(True, alpha=0.3)
            axes[1].set_title("Trend Component", fontweight="bold")

            decomposition.seasonal.plot(
                ax=axes[2],
                title="Seasonal Component",
                color=COLORS["accent1"],
                linewidth=2,
            )
            axes[2].grid(True, alpha=0.3)
            axes[2].set_title("Seasonal Component", fontweight="bold")

            decomposition.resid.plot(
                ax=axes[3],
                title="Residual Component",
                color=COLORS["accent2"],
                linewidth=2,
            )
            axes[3].grid(True, alpha=0.3)
            axes[3].set_title("Residual Component", fontweight="bold")

            plt.tight_layout()
            plt.show()

            # Calculate component strengths
            trend_strength = abs(
                decomposition.trend.dropna().std() / decomposition.observed.std()
            )
            seasonal_strength = abs(
                decomposition.seasonal.std() / decomposition.observed.std()
            )
            residual_strength = abs(
                decomposition.resid.dropna().std() / decomposition.observed.std()
            )

            print(f"Trend strength: {trend_strength:.3f}")
            print(f"Seasonal strength: {seasonal_strength:.3f}")
            print(f"Residual noise: {residual_strength:.3f}")

        except Exception as e:
            print(f"Decomposition failed: {e}")
            decomposition = None

        # 2. Stationarity Testing
        print("\n2. STATIONARITY ANALYSIS:")

        # ADF test on original series
        print("Original Series ADF Test:")
        adf_result = adfuller(ts_series.dropna())
        self._print_adf_results(adf_result)

        # ADF test on differenced series
        print("\nFirst Differenced Series ADF Test:")
        ts_diff = ts_series.diff().dropna()
        adf_result_diff = adfuller(ts_diff)
        self._print_adf_results(adf_result_diff)

        # Plot stationarity comparison
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        axes[0].plot(
            ts_series.index, ts_series.values, color=COLORS["primary"], linewidth=2
        )
        axes[0].set_title("Original Time Series", fontweight="bold")
        axes[0].set_ylabel("Weekly Sales ($)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            ts_diff.index, ts_diff.values, color=COLORS["secondary"], linewidth=2
        )
        axes[1].set_title("First Differenced Series", fontweight="bold")
        axes[1].set_ylabel("Differenced Sales ($)")
        axes[1].set_xlabel("Date")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 3. Autocorrelation Analysis
        print("\n3. AUTOCORRELATION ANALYSIS:")

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        plot_acf(
            ts_series.dropna(),
            ax=axes[0],
            lags=52,
            title="Autocorrelation Function (ACF)",
            color=COLORS["primary"],
        )
        axes[0].grid(True, alpha=0.3)

        plot_pacf(
            ts_series.dropna(),
            ax=axes[1],
            lags=52,
            title="Partial Autocorrelation Function (PACF)",
            color=COLORS["secondary"],
        )
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Calculate and report significant lags
        acf_values = acf(ts_series.dropna(), nlags=52)
        significant_lags = [
            (i, val) for i, val in enumerate(acf_values[1:], 1) if abs(val) > 0.1
        ]

        print("Significant Autocorrelation Lags (>0.1):")
        for lag, value in significant_lags[:10]:
            print(f"Lag {lag}: {value:.3f}")

        return {
            "decomposition": decomposition,
            "adf_original": adf_result,
            "adf_differenced": adf_result_diff,
            "acf_values": acf_values,
            "significant_lags": significant_lags,
        }

    def _print_adf_results(self, adf_result):
        """Helper function to print ADF test results"""
        print(f"   ADF Statistic: {adf_result[0]:.4f}")
        print(f"   p-value: {adf_result[1]:.4f}")
        print("   Critical Values:")
        for key, value in adf_result[4].items():
            print(f"      {key}: {value:.3f}")

        if adf_result[1] <= 0.05:
            print("   Result: Time series IS STATIONARY (reject null hypothesis)")
        else:
            print(
                "   Result: Time series IS NON-STATIONARY (fail to reject null hypothesis)"
            )

    def enhanced_visualizations(self):
        """Create enhanced business and technical visualizations"""
        if self.merged_data is None:
            print("Please run merge_datasets() first")
            return

        print("\n" + "=" * 60)
        print("ENHANCED BUSINESS VISUALIZATIONS")
        print("=" * 60)

        # Create comprehensive visualization suite
        fig = plt.figure(figsize=(20, 24))

        # 1. Overall sales trend over time
        ax1 = plt.subplot(4, 3, 1)
        weekly_sales = (
            self.merged_data.groupby("Date")["Weekly_Sales"].sum().reset_index()
        )
        ax1.plot(
            weekly_sales["Date"],
            weekly_sales["Weekly_Sales"],
            color=COLORS["primary"],
            linewidth=2,
        )
        ax1.set_title("Total Weekly Sales Over Time", fontweight="bold")
        ax1.set_ylabel("Total Sales ($)")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)

        # 2. Sales distribution (log scale)
        ax2 = plt.subplot(4, 3, 2)
        ax2.hist(
            self.merged_data["Weekly_Sales"],
            bins=50,
            alpha=0.8,
            color=COLORS["secondary"],
            edgecolor="white",
            linewidth=0.5,
        )
        ax2.set_title("Sales Distribution (Log Scale)", fontweight="bold")
        ax2.set_xlabel("Weekly Sales ($)")
        ax2.set_ylabel("Frequency")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

        # 3. Store type performance
        ax3 = plt.subplot(4, 3, 3)
        store_type_sales = self.merged_data.groupby("Type")["Weekly_Sales"].mean()
        bars = ax3.bar(
            store_type_sales.index,
            store_type_sales.values,
            color=COLORS["palette"][: len(store_type_sales)],
            alpha=0.8,
        )
        ax3.set_title("Average Sales by Store Type", fontweight="bold")
        ax3.set_ylabel("Average Weekly Sales ($)")
        ax3.grid(True, alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"${height:,.0f}",
                ha="center",
                va="bottom",
            )

        # 4. Holiday impact comparison
        ax4 = plt.subplot(4, 3, 4)
        holiday_sales = self.merged_data.groupby("IsHoliday")["Weekly_Sales"].mean()
        colors = [COLORS["primary"], COLORS["accent2"]]
        bars = ax4.bar(
            ["Non-Holiday", "Holiday"], holiday_sales.values, color=colors, alpha=0.8
        )
        ax4.set_title("Holiday vs Non-Holiday Sales", fontweight="bold")
        ax4.set_ylabel("Average Weekly Sales ($)")
        ax4.grid(True, alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"${height:,.0f}",
                ha="center",
                va="bottom",
            )

        # 5. Seasonal patterns
        ax5 = plt.subplot(4, 3, 5)
        self.merged_data["Month"] = self.merged_data["Date"].dt.month
        monthly_sales = self.merged_data.groupby("Month")["Weekly_Sales"].mean()
        ax5.plot(
            monthly_sales.index,
            monthly_sales.values,
            marker="o",
            linewidth=2,
            markersize=8,
            color=COLORS["accent1"],
        )
        ax5.set_title("Seasonal Sales Patterns", fontweight="bold")
        ax5.set_xlabel("Month")
        ax5.set_ylabel("Average Weekly Sales ($)")
        ax5.set_xticks(range(1, 13))
        ax5.grid(True, alpha=0.3)

        # 6. Top departments performance
        ax6 = plt.subplot(4, 3, 6)
        dept_sales = (
            self.merged_data.groupby("Dept")["Weekly_Sales"]
            .mean()
            .sort_values(ascending=True)
            .tail(10)
        )
        ax6.barh(
            range(len(dept_sales)),
            dept_sales.values,
            color=COLORS["success"],
            alpha=0.8,
        )
        ax6.set_yticks(range(len(dept_sales)))
        ax6.set_yticklabels(dept_sales.index)
        ax6.set_title("Top 10 Departments", fontweight="bold")
        ax6.set_xlabel("Average Weekly Sales ($)")
        ax6.grid(True, alpha=0.3)

        # 7. Store size vs performance scatter
        ax7 = plt.subplot(4, 3, 7)
        store_perf = self.merged_data.groupby("Store").agg(
            {"Weekly_Sales": "mean", "Size": "first", "Type": "first"}
        )

        colors_map = {
            "A": COLORS["accent2"],
            "B": COLORS["primary"],
            "C": COLORS["success"],
        }
        for store_type in ["A", "B", "C"]:
            type_data = store_perf[store_perf["Type"] == store_type]
            ax7.scatter(
                type_data["Size"],
                type_data["Weekly_Sales"],
                c=colors_map[store_type],
                label=f"Type {store_type}",
                alpha=0.7,
                s=50,
            )

        ax7.set_xlabel("Store Size (sq ft)")
        ax7.set_ylabel("Average Weekly Sales ($)")
        ax7.set_title("Store Size vs Performance", fontweight="bold")
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. Economic factors correlation
        ax8 = plt.subplot(4, 3, 8)
        econ_factors = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
        econ_corr = (
            self.merged_data[econ_factors + ["Weekly_Sales"]]
            .corr()["Weekly_Sales"]
            .drop("Weekly_Sales")
        )

        colors = [
            COLORS["success"] if x > 0 else COLORS["accent2"] for x in econ_corr.values
        ]
        bars = ax8.barh(econ_corr.index, econ_corr.values, color=colors, alpha=0.8)
        ax8.set_title("Economic Factors Correlation", fontweight="bold")
        ax8.set_xlabel("Correlation Coefficient")
        ax8.grid(True, alpha=0.3)
        ax8.axvline(x=0, color="black", linestyle="-", alpha=0.8)

        # 9. Department performance heatmap
        ax9 = plt.subplot(4, 3, 9)
        top_depts = (
            self.merged_data.groupby("Dept")["Weekly_Sales"].mean().nlargest(15).index
        )
        top_stores = (
            self.merged_data.groupby("Store")["Weekly_Sales"].sum().nlargest(8).index
        )

        heatmap_data = (
            self.merged_data[
                (self.merged_data["Dept"].isin(top_depts))
                & (self.merged_data["Store"].isin(top_stores))
            ]
            .groupby(["Store", "Dept"])["Weekly_Sales"]
            .mean()
            .unstack(fill_value=0)
        )

        sns.heatmap(
            heatmap_data, ax=ax9, cmap="viridis", cbar=True, cbar_kws={"shrink": 0.8}
        )
        ax9.set_title("Top Stores vs Departments", fontweight="bold")
        ax9.set_xlabel("Department")
        ax9.set_ylabel("Store")

        # 10. Holiday sales over time
        ax10 = plt.subplot(4, 3, 10)
        holiday_time = (
            self.merged_data.groupby(["Date", "IsHoliday"])["Weekly_Sales"]
            .sum()
            .unstack(fill_value=0)
        )

        if 0 in holiday_time.columns and 1 in holiday_time.columns:
            ax10.plot(
                holiday_time.index,
                holiday_time[0],
                label="Non-Holiday",
                color=COLORS["primary"],
                alpha=0.7,
                linewidth=2,
            )
            ax10.plot(
                holiday_time.index,
                holiday_time[1],
                label="Holiday",
                color=COLORS["accent2"],
                alpha=0.7,
                linewidth=2,
            )
            ax10.set_title("Holiday vs Non-Holiday Over Time", fontweight="bold")
            ax10.set_ylabel("Total Sales ($)")
            ax10.legend()
            ax10.tick_params(axis="x", rotation=45)
            ax10.grid(True, alpha=0.3)

        # 11. Markdown impact analysis
        ax11 = plt.subplot(4, 3, 11)
        markdown_data = self.merged_data[self.merged_data["MarkDown1"].notna()]
        if not markdown_data.empty:
            markdown_monthly = markdown_data.groupby(
                markdown_data["Date"].dt.to_period("M")
            ).agg({"MarkDown1": "mean", "Weekly_Sales": "mean"})

            ax11_twin = ax11.twinx()
            line1 = ax11.plot(
                range(len(markdown_monthly)),
                markdown_monthly["MarkDown1"],
                "b-",
                label="MarkDown1",
                linewidth=2,
                color=COLORS["warning"],
            )
            line2 = ax11_twin.plot(
                range(len(markdown_monthly)),
                markdown_monthly["Weekly_Sales"],
                "r-",
                label="Sales",
                linewidth=2,
                color=COLORS["accent2"],
            )

            ax11.set_xlabel("Time Period")
            ax11.set_ylabel("Average MarkDown1 ($)", color=COLORS["warning"])
            ax11_twin.set_ylabel("Average Sales ($)", color=COLORS["accent2"])
            ax11.set_title("MarkDown vs Sales Over Time", fontweight="bold")
            ax11.grid(True, alpha=0.3)

            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax11.legend(lines, labels, loc="upper left")

        # 12. Store type distribution pie chart
        ax12 = plt.subplot(4, 3, 12)
        store_type_dist = self.stores_data["Type"].value_counts()
        colors = COLORS["palette"][: len(store_type_dist)]
        wedges, texts, autotexts = ax12.pie(
            store_type_dist.values,
            labels=store_type_dist.index,
            autopct="%1.1f%%",
            colors=colors,
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
        ax12.set_title("Store Type Distribution", fontweight="bold")

        plt.tight_layout()
        plt.show()

    def comprehensive_correlation_analysis(self):
        """Enhanced correlation analysis with business insights"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE CORRELATION ANALYSIS")
        print("=" * 60)

        if self.merged_data is None:
            print("Please run merge_datasets() first")
            return

        # Select numeric columns for correlation
        numeric_cols = [
            "Weekly_Sales",
            "Temperature",
            "Fuel_Price",
            "CPI",
            "Unemployment",
            "Size",
            "IsHoliday",
            "Store",
            "Dept",
        ]

        # Add markdown columns if available
        markdown_cols = [col for col in self.merged_data.columns if "MarkDown" in col]
        numeric_cols.extend(markdown_cols)

        # Filter to existing columns
        available_cols = [
            col for col in numeric_cols if col in self.merged_data.columns
        ]

        # Fill missing values for correlation analysis
        corr_data = self.merged_data[available_cols].fillna(
            self.merged_data[available_cols].mean()
        )

        # Calculate correlation matrix
        correlation_matrix = corr_data.corr()

        # Create enhanced correlation visualization
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Full correlation heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            ax=axes[0],
            cmap="RdBu_r",
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
        )
        axes[0].set_title(
            "Comprehensive Correlation Matrix", fontweight="bold", fontsize=14
        )

        # Sales-focused correlation bar chart
        sales_corr = (
            correlation_matrix["Weekly_Sales"]
            .drop("Weekly_Sales")
            .sort_values(key=abs, ascending=True)
        )

        colors = [
            COLORS["success"] if x > 0 else COLORS["accent2"] for x in sales_corr.values
        ]
        bars = axes[1].barh(
            range(len(sales_corr)), sales_corr.values, color=colors, alpha=0.8
        )
        axes[1].set_yticks(range(len(sales_corr)))
        axes[1].set_yticklabels(sales_corr.index)
        axes[1].set_title(
            "Features Correlation with Sales", fontweight="bold", fontsize=14
        )
        axes[1].set_xlabel("Correlation Coefficient")
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(x=0, color="black", linestyle="-", alpha=0.8)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[1].text(
                width + (0.01 if width > 0 else -0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                ha="left" if width > 0 else "right",
                va="center",
            )

        plt.tight_layout()
        plt.show()

        # Print detailed correlation insights
        print("\nCORRELATION INSIGHTS:")
        print("Strong correlations with Weekly Sales (|r| > 0.1):")

        strong_correlations = []
        for var, corr in sales_corr.items():
            if abs(corr) > 0.1:
                direction = "positive" if corr > 0 else "negative"
                strength = (
                    "strong"
                    if abs(corr) > 0.5
                    else "moderate" if abs(corr) > 0.3 else "weak"
                )
                strong_correlations.append((var, corr, direction, strength))
                print(f"   • {var}: {corr:.3f} ({strength} {direction})")

        if not strong_correlations:
            print("   • No strong correlations found (threshold: |r| > 0.1)")

        return {
            "correlation_matrix": correlation_matrix,
            "sales_correlations": sales_corr,
            "strong_correlations": strong_correlations,
        }

    def business_insights_summary(self):
        """Generate comprehensive business insights and recommendations"""
        print("\n" + "=" * 80)
        print("BUSINESS INSIGHTS AND RECOMMENDATIONS")
        print("=" * 80)

        if self.merged_data is None:
            print("Please run merge_datasets() first")
            return

        insights = {}

        # 1. Sales Performance Metrics
        total_sales = self.merged_data["Weekly_Sales"].sum()
        avg_weekly_sales = self.merged_data["Weekly_Sales"].mean()
        num_stores = self.merged_data["Store"].nunique()
        num_departments = self.merged_data["Dept"].nunique()
        date_range = f"{self.merged_data['Date'].min().strftime('%Y-%m-%d')} to {self.merged_data['Date'].max().strftime('%Y-%m-%d')}"

        insights["performance"] = {
            "total_sales": total_sales,
            "avg_weekly_sales": avg_weekly_sales,
            "num_stores": num_stores,
            "num_departments": num_departments,
            "date_range": date_range,
        }

        # 2. Holiday Impact Analysis
        holiday_analysis = self.merged_data.groupby("IsHoliday")["Weekly_Sales"].agg(
            ["mean", "count"]
        )
        holiday_boost = (
            (holiday_analysis.loc[True, "mean"] - holiday_analysis.loc[False, "mean"])
            / holiday_analysis.loc[False, "mean"]
        ) * 100
        insights["holiday_boost"] = holiday_boost

        # 3. Top Performers
        top_store = self.merged_data.groupby("Store")["Weekly_Sales"].sum().idxmax()
        top_dept = self.merged_data.groupby("Dept")["Weekly_Sales"].sum().idxmax()
        best_store_type = (
            self.merged_data.groupby("Type")["Weekly_Sales"].mean().idxmax()
        )

        insights["top_performers"] = {
            "store": top_store,
            "department": top_dept,
            "store_type": best_store_type,
        }

        # 4. Seasonal Patterns
        monthly_sales = self.merged_data.groupby(self.merged_data["Date"].dt.month)[
            "Weekly_Sales"
        ].mean()
        best_month = monthly_sales.idxmax()
        worst_month = monthly_sales.idxmin()
        seasonal_variation = (
            (monthly_sales.max() - monthly_sales.min()) / monthly_sales.mean() * 100
        )

        insights["seasonality"] = {
            "best_month": best_month,
            "worst_month": worst_month,
            "seasonal_variation_pct": seasonal_variation,
        }

        # 5. Print Business Summary
        print(
            f"""
            BUSINESS OVERVIEW:
            • Dataset Period: {date_range}
            • Total Sales: ${total_sales:,.2f}
            • Average Weekly Sales: ${avg_weekly_sales:,.2f}
            • Number of Stores: {num_stores}
            • Number of Departments: {num_departments}

            TOP PERFORMERS:
            • Best Store: Store #{top_store}
            • Best Department: Department #{top_dept}
            • Best Store Type: Type {best_store_type}

            SEASONAL INSIGHTS:
            • Best Month: {best_month} (${monthly_sales[best_month]:,.2f} avg)
            • Worst Month: {worst_month} (${monthly_sales[worst_month]:,.2f} avg)
            • Seasonal Variation: {seasonal_variation:.1f}%

            HOLIDAY IMPACT:
            • Sales Boost During Holidays: +{holiday_boost:.1f}%
            • Holiday weeks are critical revenue periods

            KEY RECOMMENDATIONS:
            1. HOLIDAY STRATEGY:
                • Increase inventory 2-3 weeks before major holidays
                • Plan promotional campaigns around holiday periods
                • Optimize staffing for holiday demand spikes

            2. STORE OPTIMIZATION:
                • Replicate successful strategies from Store #{top_store}
                • Focus on Type {best_store_type} store format advantages
                • Investigate underperforming stores for improvement

            3. SEASONAL PLANNING:
                • Prepare for {seasonal_variation:.1f}% seasonal variation
                • Build inventory for month {best_month} demand peaks
                • Implement cost controls during month {worst_month} lows

            4. DEPARTMENT FOCUS:
                • Invest in Department #{top_dept} expansion
                • Cross-sell opportunities with high-performing departments
                • Review underperforming department strategies

            5. FORECASTING IMPROVEMENTS:
                • Incorporate holiday multipliers in predictions
                • Account for seasonal patterns in models
                • Use external economic factors for accuracy
                    """
        )

        return insights

    def run_comprehensive_analysis(self):
        """Execute the complete comprehensive EDA pipeline"""
        print("STARTING COMPREHENSIVE WALMART SALES EDA")
        print("=" * 80)

        # 1. Data Loading (if not provided in constructor)
        if self.train_data is None:
            if not self.load_data():
                return

        # 2. Data Merging
        self.merge_datasets()

        # 3. Basic Information and Quality Assessment
        self.basic_info_and_quality()

        # 4. Missing Values Analysis
        self.missing_values_comprehensive()

        # 5. Competition-Specific Analysis
        competition_results = self.competition_specific_analysis()

        # 6. Advanced Time Series Analysis
        ts_results = self.advanced_time_series_analysis()

        # 7. Enhanced Visualizations
        self.enhanced_visualizations()

        # 8. Comprehensive Correlation Analysis
        correlation_results = self.comprehensive_correlation_analysis()

        # 9. Business Insights Summary
        business_insights = self.business_insights_summary()

        print("\nCOMPREHENSIVE ANALYSIS COMPLETE!")
        print("=" * 80)
        print("Analysis Results Summary:")
        print(
            f"• Holiday Multiplier: {competition_results.get('holiday_multiplier', 'N/A'):.3f}x"
        )
        print(
            f"• Seasonal Variation: {business_insights.get('seasonality', {}).get('seasonal_variation_pct', 'N/A'):.1f}%"
        )
        print(
            f"• Top Store: #{business_insights.get('top_performers', {}).get('store', 'N/A')}"
        )
        print(
            f"• Top Department: #{business_insights.get('top_performers', {}).get('department', 'N/A')}"
        )

        return {
            "competition_analysis": competition_results,
            "time_series_analysis": ts_results,
            "correlation_analysis": correlation_results,
            "business_insights": business_insights,
        }


if __name__ == "__main__":
    # Initialize with data (if available) or let it load from files
    eda = WalmartComprehensiveEDA()

    # Run complete comprehensive analysis
    results = eda.run_comprehensive_analysis()

    print("\nComprehensive EDA completed successfully!")
    print("Results available in the 'results' dictionary")
