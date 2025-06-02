import pandas as pd
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

try:
    HTS_AVAILABLE = False
except ImportError:
    HTS_AVAILABLE = False


class HierarchicalTimeSeriesModel:
    """Advanced forecasting models for Walmart competition including hierarchical and causal approaches"""

    def __init__(self, data):
        self.data = data
        self.models = {}
        self.results = {}
        self.hierarchical_structure = None
        self.feature_columns = []
        self.train_data = None
        self.val_data = None

    def prepare_hierarchical_data(self, validation_weeks=8):
        """Prepare data for hierarchical time series modeling using natural Store->Department structure"""
        print("=== PREPARING HIERARCHICAL DATA STRUCTURE ===")

        self.data_clean = self.data.dropna(subset=["Weekly_Sales"])
        self.data_clean = self.data_clean.sort_values(["Store", "Dept", "Date"])

        stores = sorted(self.data_clean["Store"].unique())
        store_dept_combos = (
            self.data_clean.groupby(["Store", "Dept"]).size().index.tolist()
        )

        self.hierarchical_structure = {
            "Total": stores,
            **{
                f"Store_{store}": [
                    f"Store_{store}_Dept_{dept}"
                    for store_inner, dept in store_dept_combos
                    if store_inner == store
                ]
                for store in stores
            },
        }

        unique_dates = sorted(self.data_clean["Date"].unique())
        split_date = unique_dates[-validation_weeks]

        self.train_data = self.data_clean[self.data_clean["Date"] < split_date].copy()
        self.val_data = self.data_clean[self.data_clean["Date"] >= split_date].copy()

        exclude_cols = ["Weekly_Sales", "Store", "Dept", "Date"]
        self.feature_columns = [
            col
            for col in self.data_clean.columns
            if col not in exclude_cols and not col.endswith("_scaled")
        ]

        print(f"Natural hierarchy structure created:")
        print(f"  - Level 0 (Total): 1 node")
        print(f"  - Level 1 (Stores): {len(stores)} nodes")
        print(f"  - Level 2 (Store-Dept): {len(store_dept_combos)} nodes")
        print(f"Training data: {self.train_data.shape}")
        print(f"Validation data: {self.val_data.shape}")

        return self.train_data, self.val_data

    def hierarchical_time_series_model(self):
        """Hierarchical Time Series Forecasting using natural Store->Department structure"""
        print("=== TRAINING HIERARCHICAL TIME SERIES MODEL ===")
        start_time = time.time()

        if not HTS_AVAILABLE:
            print("HTS library not available. Using custom reconciliation approach.")
            return self._custom_hierarchical_reconciliation()

        try:
            stores = sorted(self.train_data["Store"].unique())
            store_dept_combos = (
                self.train_data.groupby(["Store", "Dept"]).size().index.tolist()
            )
            dates = sorted(self.train_data["Date"].unique())

            columns = (
                ["Total"]
                + [f"Store_{store}" for store in stores]
                + [f"Store_{store}_Dept_{dept}" for store, dept in store_dept_combos]
            )

            hierarchy_data = []

            for date in dates:
                date_data = self.train_data[self.train_data["Date"] == date]
                row = []

                total_sales = date_data["Weekly_Sales"].sum()
                row.append(total_sales)

                for store in stores:
                    store_sales = date_data[date_data["Store"] == store][
                        "Weekly_Sales"
                    ].sum()
                    row.append(store_sales)

                for store, dept in store_dept_combos:
                    store_dept_sales = date_data[
                        (date_data["Store"] == store) & (date_data["Dept"] == dept)
                    ]["Weekly_Sales"].sum()
                    row.append(store_dept_sales)

                hierarchy_data.append(row)

            hierarchy_df = pd.DataFrame(hierarchy_data, columns=columns, index=dates)
            hierarchy_df = hierarchy_df.fillna(0)

            print(f"Hierarchy matrix shape: {hierarchy_df.shape}")

            hierarchy_dict = {"Total": [f"Store_{store}" for store in stores]}

            for store in stores:
                store_depts = [
                    f"Store_{store}_Dept_{dept}"
                    for s, dept in store_dept_combos
                    if s == store
                ]
                if store_depts:
                    hierarchy_dict[f"Store_{store}"] = store_depts

            model = HTSRegressor(model="linear", revision_method="OLS", n_jobs=1)
            model.fit(hierarchy_df.values)

            val_dates = sorted(self.val_data["Date"].unique())
            predictions = model.predict(steps_ahead=len(val_dates))

            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)

            total_predictions = predictions[:, 0]

            actual_totals = []
            for date in val_dates:
                date_data = self.val_data[self.val_data["Date"] == date]
                actual_totals.append(date_data["Weekly_Sales"].sum())

            training_time = time.time() - start_time

            self.models["HTS"] = model
            self.results["HTS"] = {
                "predictions": total_predictions[: len(actual_totals)],
                "actual": np.array(actual_totals),
                "weights": np.ones(len(actual_totals)),
                "training_time": training_time,
                "model_type": "Hierarchical",
                "hierarchy_structure": hierarchy_dict,
                "hierarchy_matrix": hierarchy_df,
                "all_predictions": predictions,
            }

            print(
                f"Hierarchical Time Series model trained in {training_time:.2f} seconds"
            )

            return model, total_predictions[: len(actual_totals)]

        except Exception as e:
            print(f"HTS library method failed: {e}")
            print("Falling back to custom hierarchical reconciliation...")
            return self._custom_hierarchical_reconciliation()

    def _custom_hierarchical_reconciliation(self):
        """Enhanced OLS hierarchical reconciliation method"""
        print("=== OLS HIERARCHICAL RECONCILIATION ===")
        start_time = time.time()

        try:
            stores = sorted(self.train_data["Store"].unique())
            store_dept_combos = (
                self.train_data.groupby(["Store", "Dept"]).size().index.tolist()
            )

            dates_train = sorted(self.train_data["Date"].unique())
            dates_val = sorted(self.val_data["Date"].unique())

            hierarchy_ts = self._build_hierarchy_matrix(
                dates_train, stores, store_dept_combos, self.train_data
            )

            base_forecasts = self._generate_comprehensive_base_forecasts(
                hierarchy_ts, len(dates_val), stores, store_dept_combos
            )

            S = self._create_summing_matrix(stores, store_dept_combos)

            reconciled_forecasts = self._apply_ols_reconciliation(
                base_forecasts, S, len(dates_val), stores, store_dept_combos
            )

            actual_totals = []
            for date in dates_val:
                date_data = self.val_data[self.val_data["Date"] == date]
                actual_totals.append(date_data["Weekly_Sales"].sum())

            val_weights = self._calculate_holiday_weights(dates_val)

            training_time = time.time() - start_time

            self.models["HTS"] = "OLS_Reconciliation"
            self.results["HTS"] = {
                "predictions": reconciled_forecasts["Total"],
                "actual": np.array(actual_totals),
                "weights": val_weights,
                "training_time": training_time,
                "model_type": "Hierarchical",
                "reconciliation_method": "OLS",
                "hierarchy_structure": {
                    "Total": [f"Store_{s}" for s in stores],
                    **{
                        f"Store_{s}": [
                            f"Store_{s}_Dept_{d}"
                            for ss, d in store_dept_combos
                            if ss == s
                        ]
                        for s in stores
                    },
                },
                "all_forecasts": reconciled_forecasts,
                "hierarchy_ts": hierarchy_ts,
                "summing_matrix": S,
            }

            wmae = np.sum(
                val_weights
                * np.abs(np.array(actual_totals) - reconciled_forecasts["Total"])
            ) / np.sum(val_weights)
            mae = np.mean(
                np.abs(np.array(actual_totals) - reconciled_forecasts["Total"])
            )

            print(
                f"OLS hierarchical reconciliation completed in {training_time:.2f} seconds"
            )
            print(f"Validation WMAE: {wmae:.2f}")
            print(f"Validation MAE: {mae:.2f}")

            return "OLS_Reconciliation", reconciled_forecasts["Total"]

        except Exception as e:
            print(f"OLS hierarchical reconciliation failed: {e}")
            print("Falling back to bottom-up reconciliation...")
            return self._fallback_bottom_up_reconciliation()

    def _build_hierarchy_matrix(self, dates, stores, store_dept_combos, data):
        """Build complete hierarchy matrix for all levels"""
        hierarchy_ts = {}

        total_ts = []
        for date in dates:
            date_data = data[data["Date"] == date]
            total_ts.append(date_data["Weekly_Sales"].sum())
        hierarchy_ts["Total"] = np.array(total_ts)

        for store in stores:
            store_ts = []
            for date in dates:
                date_data = data[(data["Date"] == date) & (data["Store"] == store)]
                store_ts.append(date_data["Weekly_Sales"].sum())
            hierarchy_ts[f"Store_{store}"] = np.array(store_ts)

        for store, dept in store_dept_combos:
            store_dept_ts = []
            for date in dates:
                date_data = data[
                    (data["Date"] == date)
                    & (data["Store"] == store)
                    & (data["Dept"] == dept)
                ]
                store_dept_ts.append(date_data["Weekly_Sales"].sum())
            hierarchy_ts[f"Store_{store}_Dept_{dept}"] = np.array(store_dept_ts)

        return hierarchy_ts

    def _generate_comprehensive_base_forecasts(
        self, hierarchy_ts, forecast_horizon, stores, store_dept_combos
    ):
        """Generate base forecasts for all hierarchy levels"""
        forecasts = {}

        hierarchy_nodes = (
            ["Total"]
            + [f"Store_{store}" for store in stores]
            + [f"Store_{store}_Dept_{dept}" for store, dept in store_dept_combos]
        )

        for i, key in enumerate(hierarchy_nodes):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(hierarchy_nodes)} series forecasted")

            ts = hierarchy_ts[key]

            if len(ts) < 4:
                forecasts[key] = np.full(forecast_horizon, max(ts.mean(), 0))
                continue

            try:
                forecast_trend = self._forecast_trend_seasonal(ts, forecast_horizon)
                forecast_exp = self._forecast_exponential_smoothing(
                    ts, forecast_horizon
                )
                forecast_ma = self._forecast_moving_average_trend(ts, forecast_horizon)

                forecast_ensemble = (
                    0.5 * forecast_trend + 0.3 * forecast_exp + 0.2 * forecast_ma
                )
                forecasts[key] = np.maximum(forecast_ensemble, 0)

            except Exception as e:
                forecasts[key] = self._forecast_trend_seasonal(ts, forecast_horizon)

        return forecasts

    def _forecast_trend_seasonal(self, ts, horizon):
        """Simple trend + seasonal forecasting"""
        if len(ts) < 8:
            return np.full(horizon, max(ts.mean(), 0))

        recent_trend = (ts[-4:].mean() - ts[-8:-4].mean()) / 4

        if len(ts) >= 52:
            seasonal = ts[-52:] - np.mean(ts[-52:])
            seasonal_cycle = (
                seasonal[-horizon:]
                if horizon <= 52
                else np.tile(seasonal, (horizon // 52) + 1)[:horizon]
            )
        else:
            seasonal_cycle = np.zeros(horizon)

        base_level = ts[-4:].mean()

        forecasts = []
        for i in range(horizon):
            forecast_val = base_level + recent_trend * (i + 1) + seasonal_cycle[i]
            forecasts.append(forecast_val)

        return np.array(forecasts)

    def _forecast_exponential_smoothing(self, ts, horizon, alpha=0.3):
        """Simple exponential smoothing"""
        if len(ts) == 0:
            return np.zeros(horizon)

        s = ts[0]
        for val in ts[1:]:
            s = alpha * val + (1 - alpha) * s

        return np.full(horizon, max(s, 0))

    def _forecast_moving_average_trend(self, ts, horizon, window=4):
        """Moving average with trend"""
        if len(ts) < window * 2:
            return np.full(horizon, max(ts.mean(), 0))

        ma_recent = ts[-window:].mean()
        ma_older = ts[-2 * window : -window].mean()
        trend = (ma_recent - ma_older) / window

        forecasts = []
        for i in range(horizon):
            forecast_val = ma_recent + trend * (i + 1)
            forecasts.append(forecast_val)

        return np.array(forecasts)

    def _create_summing_matrix(self, stores, store_dept_combos):
        """Create summing matrix S where: y = S * b"""
        n_bottom = len(store_dept_combos)
        n_middle = len(stores)
        n_total = 1
        n_all = n_total + n_middle + n_bottom

        S = np.zeros((n_all, n_bottom))
        row_idx = 0

        S[row_idx, :] = 1
        row_idx += 1

        for i, store in enumerate(stores):
            for j, (s, d) in enumerate(store_dept_combos):
                if s == store:
                    S[row_idx, j] = 1
            row_idx += 1

        S[row_idx:, :] = np.eye(n_bottom)

        return S

    def _apply_ols_reconciliation(
        self, base_forecasts, S, forecast_horizon, stores, store_dept_combos
    ):
        """Apply OLS reconciliation: reconciled = S * (S'S)^(-1) * S' * base_forecasts"""

        hierarchy_nodes = (
            ["Total"]
            + [f"Store_{store}" for store in stores]
            + [f"Store_{store}_Dept_{dept}" for store, dept in store_dept_combos]
        )

        n_series = len(hierarchy_nodes)
        base_forecast_matrix = np.zeros((n_series, forecast_horizon))

        for i, node in enumerate(hierarchy_nodes):
            base_forecast_matrix[i, :] = base_forecasts[node]

        try:
            StS = S.T @ S
            StS_inv = np.linalg.pinv(StS)
            reconciliation_matrix = S @ StS_inv @ S.T
            reconciled_matrix = reconciliation_matrix @ base_forecast_matrix

            reconciled_forecasts = {}
            for i, node in enumerate(hierarchy_nodes):
                reconciled_forecasts[node] = reconciled_matrix[i, :]

            self._verify_reconciliation(
                reconciled_forecasts, stores, store_dept_combos, forecast_horizon
            )

            return reconciled_forecasts

        except Exception as e:
            print(f"OLS reconciliation failed: {e}")
            return self._fallback_bottom_up_reconciliation_dict(
                base_forecasts, stores, store_dept_combos
            )

    def _verify_reconciliation(
        self, reconciled_forecasts, stores, store_dept_combos, horizon
    ):
        """Verify that reconciled forecasts satisfy hierarchy constraints"""
        tolerance = 1e-6

        for store in stores:
            store_forecast = reconciled_forecasts[f"Store_{store}"]
            dept_sum = np.zeros(horizon)
            store_depts = [
                f"Store_{store}_Dept_{dept}"
                for s, dept in store_dept_combos
                if s == store
            ]

            for dept_key in store_depts:
                if dept_key in reconciled_forecasts:
                    dept_sum += reconciled_forecasts[dept_key]

            max_error = np.max(np.abs(store_forecast - dept_sum))
            if max_error > tolerance:
                print(
                    f"  Warning: Store {store} constraint violated, max error: {max_error:.6f}"
                )

        total_forecast = reconciled_forecasts["Total"]
        store_sum = np.zeros(horizon)

        for store in stores:
            store_sum += reconciled_forecasts[f"Store_{store}"]

        max_total_error = np.max(np.abs(total_forecast - store_sum))
        if max_total_error > tolerance:
            print(
                f"  Warning: Total constraint violated, max error: {max_total_error:.6f}"
            )

    def _calculate_holiday_weights(self, dates):
        """Calculate holiday weights for validation dates"""
        weights = []
        for date in dates:
            date_data = self.val_data[self.val_data["Date"] == date]
            if len(date_data) > 0:
                is_holiday = date_data["IsHoliday"].any()
                weight = 5.0 if is_holiday else 1.0
            else:
                weight = 1.0
            weights.append(weight)

        return np.array(weights)

    def _fallback_bottom_up_reconciliation_dict(
        self, base_forecasts, stores, store_dept_combos
    ):
        """Fallback bottom-up reconciliation when OLS fails"""
        reconciled = {}
        horizon = len(next(iter(base_forecasts.values())))

        for store, dept in store_dept_combos:
            key = f"Store_{store}_Dept_{dept}"
            reconciled[key] = base_forecasts[key]

        for store in stores:
            store_depts = [
                f"Store_{store}_Dept_{dept}"
                for s, dept in store_dept_combos
                if s == store
            ]
            if store_depts:
                store_forecast = np.zeros(horizon)
                for dept_key in store_depts:
                    store_forecast += reconciled[dept_key]
                reconciled[f"Store_{store}"] = store_forecast
            else:
                reconciled[f"Store_{store}"] = base_forecasts.get(
                    f"Store_{store}", np.zeros(horizon)
                )

        total_forecast = np.zeros(horizon)
        for store in stores:
            total_forecast += reconciled[f"Store_{store}"]
        reconciled["Total"] = total_forecast

        return reconciled

    def _fallback_bottom_up_reconciliation(self):
        """Simple fallback when everything else fails"""
        print("=== FALLBACK BOTTOM-UP RECONCILIATION ===")

        stores = sorted(self.train_data["Store"].unique())
        store_dept_combos = (
            self.train_data.groupby(["Store", "Dept"]).size().index.tolist()
        )
        dates_val = sorted(self.val_data["Date"].unique())

        forecasts = {}
        for store, dept in store_dept_combos:
            store_dept_data = self.train_data[
                (self.train_data["Store"] == store) & (self.train_data["Dept"] == dept)
            ]
            if len(store_dept_data) > 0:
                avg_sales = store_dept_data["Weekly_Sales"].mean()
                forecasts[f"Store_{store}_Dept_{dept}"] = np.full(
                    len(dates_val), max(avg_sales, 0)
                )
            else:
                forecasts[f"Store_{store}_Dept_{dept}"] = np.zeros(len(dates_val))

        reconciled = self._fallback_bottom_up_reconciliation_dict(
            forecasts, stores, store_dept_combos
        )

        return "Fallback_Bottom_Up", reconciled["Total"]

    def get_all_hierarchy_predictions(self):
        """Extract predictions for all hierarchy levels after running hierarchical forecasting"""
        if "HTS" not in self.results:
            print(
                "No hierarchical forecasting results found. Run hierarchical_time_series_model() first."
            )
            return None

        all_forecasts = self.results["HTS"].get("all_forecasts", {})

        if not all_forecasts:
            print("No detailed forecasts found in results.")
            return None

        hierarchy_predictions = {"Total": {}, "Store": {}, "Store_Department": {}}
        val_dates = sorted(self.val_data["Date"].unique())

        for key, forecasts in all_forecasts.items():
            if key == "Total":
                hierarchy_predictions["Total"][key] = {
                    "forecasts": forecasts,
                    "dates": val_dates,
                    "level": "Total",
                }

            elif key.startswith("Store_") and "_Dept_" not in key:
                store_id = key.replace("Store_", "")
                hierarchy_predictions["Store"][key] = {
                    "store_id": store_id,
                    "forecasts": forecasts,
                    "dates": val_dates,
                    "level": "Store",
                }

            elif key.startswith("Store_") and "_Dept_" in key:
                parts = key.replace("Store_", "").split("_Dept_")
                store_id = parts[0]
                dept_id = parts[1]

                hierarchy_predictions["Store_Department"][key] = {
                    "store_id": store_id,
                    "department_id": dept_id,
                    "forecasts": forecasts,
                    "dates": val_dates,
                    "level": "Store_Department",
                }

        return hierarchy_predictions

    def create_prediction_dataframe(self, hierarchy_predictions=None):
        """Create a comprehensive DataFrame with all hierarchy predictions"""
        if hierarchy_predictions is None:
            hierarchy_predictions = self.get_all_hierarchy_predictions()

        if hierarchy_predictions is None:
            return None

        all_predictions = []

        for level_name, level_data in hierarchy_predictions.items():
            for entity_key, entity_data in level_data.items():
                forecasts = entity_data["forecasts"]
                dates = entity_data["dates"]

                for i, (date, forecast) in enumerate(zip(dates, forecasts)):
                    row = {
                        "Date": date,
                        "Hierarchy_Level": level_name,
                        "Entity_Key": entity_key,
                        "Forecast_Period": i + 1,
                        "Predicted_Sales": forecast,
                    }

                    if level_name == "Store":
                        row["Store_ID"] = entity_data["store_id"]
                        row["Department_ID"] = None
                    elif level_name == "Store_Department":
                        row["Store_ID"] = entity_data["store_id"]
                        row["Department_ID"] = entity_data["department_id"]
                    else:
                        row["Store_ID"] = None
                        row["Department_ID"] = None

                    all_predictions.append(row)

        predictions_df = pd.DataFrame(all_predictions)
        return predictions_df

    def get_specific_predictions(self, level="all", store_id=None, dept_id=None):
        """Get predictions for specific hierarchy levels or entities"""
        hierarchy_predictions = self.get_all_hierarchy_predictions()

        if hierarchy_predictions is None:
            return None

        val_dates = sorted(self.val_data["Date"].unique())

        if level.lower() == "total":
            total_forecasts = hierarchy_predictions["Total"]["Total"]["forecasts"]
            return pd.DataFrame(
                {"Date": val_dates, "Total_Predicted_Sales": total_forecasts}
            )

        elif level.lower() == "store":
            store_predictions = {}
            for key, data in hierarchy_predictions["Store"].items():
                if store_id is None or data["store_id"] == str(store_id):
                    store_predictions[key] = {
                        "dates": val_dates,
                        "forecasts": data["forecasts"],
                        "store_id": data["store_id"],
                    }

            if store_id is not None and len(store_predictions) == 1:
                key = list(store_predictions.keys())[0]
                return pd.DataFrame(
                    {
                        "Date": val_dates,
                        "Store_ID": store_predictions[key]["store_id"],
                        "Store_Predicted_Sales": store_predictions[key]["forecasts"],
                    }
                )

            return store_predictions

        elif level.lower() == "department":
            dept_predictions = {}
            for key, data in hierarchy_predictions["Store_Department"].items():
                include = True
                if store_id is not None and data["store_id"] != str(store_id):
                    include = False
                if dept_id is not None and data["department_id"] != str(dept_id):
                    include = False

                if include:
                    dept_predictions[key] = {
                        "dates": val_dates,
                        "forecasts": data["forecasts"],
                        "store_id": data["store_id"],
                        "department_id": data["department_id"],
                    }

            if (
                store_id is not None
                and dept_id is not None
                and len(dept_predictions) == 1
            ):
                key = list(dept_predictions.keys())[0]
                return pd.DataFrame(
                    {
                        "Date": val_dates,
                        "Store_ID": dept_predictions[key]["store_id"],
                        "Department_ID": dept_predictions[key]["department_id"],
                        "Department_Predicted_Sales": dept_predictions[key][
                            "forecasts"
                        ],
                    }
                )

            return dept_predictions

        else:
            return self.create_prediction_dataframe(hierarchy_predictions)

    def export_all_predictions(self, filename="hierarchical_predictions.csv"):
        """Export all hierarchy predictions to CSV file"""
        predictions_df = self.get_specific_predictions(level="all")

        if predictions_df is not None:
            predictions_df.to_csv(filename, index=False)
            print(f"All predictions exported to: {filename}")
            print(f"File contains {len(predictions_df)} rows")

            summary = (
                predictions_df.groupby("Hierarchy_Level")
                .agg({"Entity_Key": "nunique", "Predicted_Sales": ["mean", "sum"]})
                .round(2)
            )
            print(summary)
        else:
            print("No predictions to export")

    def print_prediction_summary(self):
        """Print a comprehensive summary of all predictions"""
        hierarchy_predictions = self.get_all_hierarchy_predictions()

        if hierarchy_predictions is None:
            return

        print("\n" + "=" * 60)
        print("HIERARCHICAL FORECASTING PREDICTION SUMMARY")
        print("=" * 60)

        val_dates = sorted(self.val_data["Date"].unique())

        total_forecasts = hierarchy_predictions["Total"]["Total"]["forecasts"]
        print(f"\nTOTAL LEVEL PREDICTIONS:")
        print(f"   Forecast periods: {len(total_forecasts)}")
        print(f"   Date range: {val_dates[0]} to {val_dates[-1]}")
        print(f"   Average weekly sales: ${total_forecasts.mean():,.0f}")
        print(f"   Total forecast: ${total_forecasts.sum():,.0f}")

        store_data = hierarchy_predictions["Store"]
        if store_data:
            print(f"\nSTORE LEVEL PREDICTIONS:")
            print(f"   Number of stores: {len(store_data)}")

            store_avgs = []
            for key, data in store_data.items():
                store_avgs.append(
                    {
                        "Store": data["store_id"],
                        "Avg_Sales": data["forecasts"].mean(),
                    }
                )

            store_avgs = sorted(store_avgs, key=lambda x: x["Avg_Sales"], reverse=True)
            print(f"   Top 5 stores by avg weekly sales:")
            for i, store in enumerate(store_avgs[:5], 1):
                print(
                    f"   {i}. Store {store['Store']}: ${store['Avg_Sales']:,.0f}/week"
                )

        dept_data = hierarchy_predictions["Store_Department"]
        if dept_data:
            print(f"\nDEPARTMENT LEVEL PREDICTIONS:")
            print(f"   Number of store-dept combinations: {len(dept_data)}")

            dept_totals = {}
            for key, data in dept_data.items():
                dept_id = data["department_id"]
                if dept_id not in dept_totals:
                    dept_totals[dept_id] = []
                dept_totals[dept_id].extend(data["forecasts"])

            dept_summary = []
            for dept_id, sales_list in dept_totals.items():
                dept_summary.append(
                    {
                        "Department": dept_id,
                        "Total_Sales": sum(sales_list),
                    }
                )

            dept_summary = sorted(
                dept_summary, key=lambda x: x["Total_Sales"], reverse=True
            )
            print(f"   Top 5 departments by total sales:")
            for i, dept in enumerate(dept_summary[:5], 1):
                print(
                    f"   {i}. Dept {dept['Department']}: ${dept['Total_Sales']:,.0f} total"
                )

    def verify_hierarchy_consistency(self):
        """Verify that predictions satisfy hierarchy constraints"""
        hierarchy_predictions = self.get_all_hierarchy_predictions()

        if hierarchy_predictions is None:
            return False

        print("\n=== VERIFYING HIERARCHY CONSISTENCY ===")

        val_dates = sorted(self.val_data["Date"].unique())
        tolerance = 1e-6

        total_preds = hierarchy_predictions["Total"]["Total"]["forecasts"]
        store_preds = {
            k: v["forecasts"] for k, v in hierarchy_predictions["Store"].items()
        }
        dept_preds = {
            k: v["forecasts"]
            for k, v in hierarchy_predictions["Store_Department"].items()
        }

        store_sum = np.zeros(len(val_dates))
        for store_forecasts in store_preds.values():
            store_sum += store_forecasts

        total_error = np.max(np.abs(total_preds - store_sum))
        if total_error < tolerance:
            print("Total = Sum of Stores: CONSISTENT")
        else:
            print(f"Total = Sum of Stores: ERROR = {total_error:.6f}")

        stores = sorted(self.train_data["Store"].unique())
        all_store_errors = []

        for store in stores:
            store_key = f"Store_{store}"
            if store_key in store_preds:
                store_forecast = store_preds[store_key]

                dept_sum = np.zeros(len(val_dates))
                for dept_key, dept_forecast in dept_preds.items():
                    if dept_key.startswith(f"Store_{store}_Dept_"):
                        dept_sum += dept_forecast

                store_error = np.max(np.abs(store_forecast - dept_sum))
                all_store_errors.append(store_error)

                if store_error > tolerance:
                    print(f"Store {store}: ERROR = {store_error:.6f}")

        if all(error < tolerance for error in all_store_errors):
            print("All Stores = Sum of Departments: CONSISTENT")
        else:
            max_error = max(all_store_errors)
            print(f"Store consistency: MAX ERROR = {max_error:.6f}")

        return total_error < tolerance and all(
            error < tolerance for error in all_store_errors
        )


if __name__ == "__main__":
    print("Hierarchical Time Series Forecasting Model Ready!")

    from src.data_loader import WalmartDataLoader
    from src.data_processing import WalmartComprehensiveEDA
    from src.feature_engineering import WalmartFeatureEngineering

    data_loader = WalmartDataLoader()
    data_loader.load_data()

    eda = WalmartComprehensiveEDA(
        data_loader.train_data,
        data_loader.test_data,
        data_loader.features_data,
        data_loader.stores_data,
    )
    merged_data = eda.merge_datasets()

    feature_eng = WalmartFeatureEngineering(merged_data)
    processed_data = feature_eng.create_walmart_features()
    processed_data = feature_eng.handle_missing_values()

    hts_model = HierarchicalTimeSeriesModel(processed_data)
    train_data, val_data = hts_model.prepare_hierarchical_data()
    model, predictions = hts_model.hierarchical_time_series_model()

    # Get all hierarchy predictions
    all_preds = hts_model.get_all_hierarchy_predictions()

    # Create prediction DataFrame
    df_all = hts_model.get_specific_predictions(level="all")

    # Print summary and verify consistency
    hts_model.print_prediction_summary()
    hts_model.verify_hierarchy_consistency()

    # Export predictions
    # hts_model.export_all_predictions("walmart_hierarchical_predictions.csv")
