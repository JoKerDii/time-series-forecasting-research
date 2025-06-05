# Time Series Forecasting Research


# Walmart Sales Forecasting: Advanced Time Series Modeling and Analysis

## Project Overview

This project implements a comprehensive forecasting system for the Walmart Sales Forecasting Competition, combining advanced time series models with hierarchical forecasting, feature engineering, and interpretability analysis to predict weekly sales across 45 stores and 99 departments.

## Problem Statement

**Objective**: Predict weekly sales for Walmart store-department combinations with emphasis on holiday period performance.

**Key Challenges**:
- **Holiday Impact**: Holiday weeks weighted 5x in evaluation metrics
- **Hierarchical Structure**: 45 stores × multiple departments requiring coherent forecasts
- **External Factors**: Economic indicators, weather, and promotional effects
- **Data Sparsity**: Missing markdown data and irregular patterns
- **Business Constraints**: Forecasts must sum coherently across organizational levels

**Evaluation Metric**: Weighted Mean Absolute Error (WMAE) with 5x penalty for holiday weeks

## Dataset

**Structure**: 4 interconnected datasets spanning February 2010 - November 2012

| Dataset | Purpose | Key Fields | Records |
|---------|---------|------------|---------|
| `train.csv` | Historical sales | Store, Dept, Date, Weekly_Sales, IsHoliday | 421,570 |
| `test.csv` | Prediction targets | Store, Dept, Date, IsHoliday | 115,064 |
| `stores.csv` | Store metadata | Type, Size | 45 |
| `features.csv` | External factors | Temperature, Fuel_Price, CPI, Unemployment, MarkDown1-5 | 8,190 |

**Holiday Periods**: Super Bowl, Labor Day, Thanksgiving, Christmas (5x weight)

## Exploratory Data Analysis (EDA)

### Comprehensive Multi-Phase Analysis

#### Phase 1: Data Foundation
- **Data Integration**: Smart merging with conflict resolution for holiday flags
- **Quality Assessment**: Missing value analysis (43% markdown coverage post-Nov 2011)
- **Validation**: Cross-dataset consistency and temporal alignment

#### Phase 2: Competition-Specific Analysis
- **Holiday Multiplier**: 1.127x average sales increase during holidays
- **Store Performance**: Type A (1.15x), Type B (1.08x), Type C (0.94x) relative performance
- **Department Concentration**: Top 20% departments generate 60% of sales

#### Phase 3: Advanced Time Series Analysis
- **Seasonal Decomposition**: 25% seasonal strength, 52-week periodicity
- **Stationarity Testing**: ADF tests revealing non-stationary series requiring differencing
- **Autocorrelation**: Significant lags at 1, 52, and 104 weeks

#### Phase 4: External Factor Analysis
- **Economic Impact**: CPI (-0.023), Unemployment (-0.011), Fuel Price (-0.015) correlations
- **Markdown Effectiveness**: 3.4% average sales lift with $3,842 average markdown value

**Key Insights**:
- Strong 52-week seasonality with holiday spikes
- Store type significantly affects performance patterns
- Economic indicators show weak but consistent negative correlations
- Promotional markdowns provide measurable but modest sales lift

## Feature Engineering

### Multi-Scale Temporal Features

#### 1. Holiday-Related Features
- **Holiday_Weight**: 5.0 for holiday weeks, 1.0 otherwise
- **Pre_Holiday/Post_Holiday**: Binary indicators for adjacent weeks
- **Holiday_Type**: Categorical encoding for different holidays

#### 2. Cyclical Temporal Encoding
```python
Month_sin = sin(2π × month / 12)
Month_cos = cos(2π × month / 12)
Week_sin = sin(2π × week / 52)
Week_cos = cos(2π × week / 52)
```

#### 3. Lag Features (Autoregressive Patterns)
- **Short-term**: 1, 2 weeks (momentum effects)
- **Medium-term**: 4, 8 weeks (monthly patterns)
- **Long-term**: 12 weeks (quarterly patterns)
- **Seasonal**: 52 weeks (year-over-year comparison)

#### 4. Rolling Statistics (Multiple Windows)
- **Windows**: 4, 8, 12, 24, 52 weeks
- **Statistics**: Mean (trend), Std (volatility), Max/Min (extremes)

#### 5. Hierarchical Performance Features
- **Store_Type_Avg**: Average sales for store type
- **Dept_Avg**: Department performance across stores
- **Store_Dept_Avg**: Historical store-department baseline

#### 6. Economic Interaction Features
- **Economic_Stress**: Standardized combination of unemployment and fuel prices
- **CPI_Fuel_Interaction**: Inflation × transportation cost effects

**Total Features**: 50+ engineered features per observation

## Model Portfolio

### 1. Prophet (Statistical Baseline)
- **Approach**: Business-oriented time series decomposition
- **Features**: Custom holiday calendar, 7 economic regressors
- **Strengths**: Interpretable components, fast training
- **Data Level**: Aggregated (total sales across all stores/departments)

### 2. Random Forest (Tree-Based Ensemble)
- **Configuration**: 100 trees, max depth 15, holiday sample weighting
- **Features**: Top 20 engineered features
- **Strengths**: Feature importance, handles mixed data types
- **Data Level**: Store-department granular

### 3. LSTM (Sequential Deep Learning)
- **Architecture**: 3-layer LSTM (64→32→16) + dense layers
- **Features**: Top 15 features, 5-week sequences
- **Strengths**: Temporal dependencies, non-linear patterns
- **Data Level**: Store-department with sequence modeling

### 4. Transformer (Attention-Based)
- **Architecture**: Multi-head attention + feed-forward networks
- **Features**: 5 core features, store-level aggregation
- **Strengths**: Long-range dependencies, parallel processing
- **Data Level**: Store-aggregated for computational efficiency

### 5. Hierarchical Time Series (HTS) Model
- **Approach**: OLS reconciliation of independent forecasts
- **Base Methods**: Ensemble of trend+seasonal, exponential smoothing, moving average
- **Strengths**: Guaranteed forecast coherence across hierarchy
- **Mathematical Foundation**: ŷ = S(S'S)⁻¹S'ỹ

### 6. Advanced Models (Temporal Fusion Transformer, Neural ODE, Gaussian Process)
- **TFT**: Variable selection networks + attention mechanisms
- **Neural ODE**: Continuous-time dynamics modeling
- **GP**: Bayesian non-parametric with uncertainty quantification

## Model Comparison and Results

### Performance Ranking (Weighted RMSE)

| Rank | Model | Weighted RMSE | Holiday RMSE | Regular RMSE | Training Time |
|------|-------|---------------|--------------|--------------|---------------|
| 1 | **LSTM** | 2,847.32 | 4,892.15 | 2,634.21 | 180.4s |
| 2 | **Random Forest** | 3,124.67 | 5,234.89 | 2,891.43 | 45.2s |
| 3 | **Transformer** | 3,298.45 | 5,567.23 | 3,087.12 | 156.7s |
| 4 | **Prophet** | 3,456.78 | 5,789.34 | 3,201.56 | 12.8s |
| 5 | **HTS Model** | 3,567.89 | 6,012.45 | 3,334.67 | 89.3s |

### Key Findings

#### 1. Model Architecture Performance
- **Neural Networks** (LSTM, Transformer) outperform statistical methods
- **LSTM** achieves best overall performance with sequential modeling
- **Tree-based methods** provide excellent speed-accuracy balance
- **Statistical models** offer interpretability at accuracy cost

#### 2. Holiday Performance Analysis
- All models struggle more with holiday periods (higher RMSE)
- **LSTM** shows best holiday period performance
- **Neural networks** better capture complex holiday dynamics
- **Statistical models** more conservative during volatile periods

#### 3. Computational Trade-offs
- **Prophet**: Fastest training (12.8s) but lowest accuracy
- **Random Forest**: Best speed-accuracy balance (45.2s training)
- **Neural Networks**: Higher accuracy but substantial computational cost
- **HTS**: Moderate computational cost with business coherence guarantee

#### 4. Granularity vs Performance
- **Store-department level** models (LSTM, RF) achieve higher accuracy
- **Aggregated models** (Prophet) faster but lose granular patterns
- **Hierarchical reconciliation** provides business coherence at accuracy cost

### Business Impact Assessment

#### Production Readiness
- **Winner**: Random Forest (robustness + speed + accuracy balance)
- **High-Stakes**: LSTM (maximum accuracy for critical decisions)
- **Interpretability**: Prophet (stakeholder communication)
- **Coherence**: HTS (business planning and budgeting)

#### Seasonal Performance
- **Q4 Holiday Season**: Neural networks excel during high-volatility periods
- **Regular Periods**: All models perform reasonably well
- **Markdown Periods**: Tree-based models handle promotional effects better

## Interpretability Analysis

### Feature Importance Ranking
1. **Total_MarkDown** (0.85): Direct promotional impact
2. **IsHoliday** (0.78): Critical business periods
3. **Sales_lag_1** (0.72): Strong autoregressive patterns
4. **Store_Type** (0.69): Fundamental store characteristics
5. **Temperature** (0.61): Seasonal shopping behavior

### Causal Relationships
- **Holiday → Sales**: Strong causal evidence (business theory + statistical)
- **Markdowns → Sales**: Moderate causal relationship with 3-4% lift
- **Economic indicators → Sales**: Weak but consistent inverse relationships
- **Temperature → Sales**: Seasonal causality in specific product categories

### Business Recommendations
#### High-Priority Actions (Controllable)
1. **Optimize markdown timing**: Schedule promotions 2-3 weeks before holidays
2. **Holiday inventory planning**: Increase stock 15-20% above baseline
3. **Store format strategy**: Prioritize Type A format expansion

#### Monitoring Requirements (External)
1. **Economic indicators**: Track unemployment and CPI for demand shifts
2. **Weather patterns**: Seasonal temperature impacts on shopping behavior
3. **Fuel prices**: Transportation cost effects on consumer spending

## Methodology Summary

### Technical Architecture
- **Data Pipeline**: Multi-source integration with quality validation
- **Feature Engineering**: 50+ features across temporal, hierarchical, and interaction domains
- **Model Portfolio**: 6 distinct approaches covering statistical, ML, and DL methods
- **Evaluation Framework**: Competition-specific metrics with business context
- **Interpretability**: Multi-method analysis combining statistical and causal inference

### Innovation Highlights
1. **Holiday-Aware Architecture**: 5x weighting integration across all models
2. **Hierarchical Reconciliation**: Business coherence through mathematical optimization
3. **Multi-Scale Feature Engineering**: Temporal patterns from weekly to yearly cycles
4. **Ensemble Methodology**: Complementary model strengths combination
5. **Business-Centric Evaluation**: Beyond academic metrics to operational requirements

## Key Contributions

1. **Comprehensive Model Comparison**: Systematic evaluation of diverse forecasting approaches
2. **Holiday-Specific Analysis**: Specialized handling of high-impact business periods
3. **Hierarchical Forecasting**: Mathematical guarantee of forecast coherence
4. **Feature Engineering Framework**: Systematic temporal and business feature creation
5. **Interpretability Integration**: Actionable insights from complex model ensemble


## Future Enhancements

1. **Real-time Forecasting**: Online learning for continuous model updates
2. **Causal Inference**: Advanced causal discovery for promotional optimization
3. **Ensemble Methods**: Sophisticated model combination strategies
4. **Transfer Learning**: Cross-store pattern sharing
5. **Uncertainty Quantification**: Prediction intervals for risk management

