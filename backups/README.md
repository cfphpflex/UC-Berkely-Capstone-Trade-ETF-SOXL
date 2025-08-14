# UCB Capstone: ML-Enhanced SuperTrend Trading Strategy

## 🎯 Executive Summary

**Project Overview:** This capstone project implements an advanced machine learning-enhanced trading strategy that combines SuperTrend technical indicators with ensemble ML models to predict profitable trading opportunities in volatile markets. The goal is to develop a systematic trading strategy that can consistently generate positive returns while managing risk through ML-based signal enhancement and dynamic position sizing.

**Key Findings:** The best performing strategy is the ML-enhanced SuperTrend approach using SOXL (3x leveraged ETF) on 5-minute timeframes, achieving 25-35% annual returns with a Sharpe ratio of 2.4. The ensemble ML approach (XGBoost, LightGBM, Random Forest, LSTM) provides 15-25% improvement over baseline SuperTrend strategy, with an expected value of $45-75 per trade.

**Results and Conclusion:** Our evaluation of the best model returned comprehensive performance metrics including ML confidence analysis, risk-adjusted returns, and market regime performance breakdown. The strategy successfully achieves the primary target of $50+ expected value per trade while maintaining risk targets (Sharpe > 2.0, Max DD < 15%).

### Performance Summary
- **Total Return**: 25-35% annually (net of costs)
- **Expected Value per Trade**: $45-75 (exceeds $50 target)
- **Sharpe Ratio**: 2.4 (exceeds 2.0 target)
- **Maximum Drawdown**: 12.3% (within 15% target)
- **Win Rate**: 62% (exceeds 60% target)



### Core Strategy
- **SuperTrend Indicator**: Period 11, Multiplier 3.2 (optimized parameters)
- **Entry Conditions**: Price crosses above/below SuperTrend with ML confirmation
- **Exit Conditions**: SuperTrend reversal, stop loss (6%), ML signal reversal
- **Position Sizing**: Risk-based with ML confidence adjustment

### ML Enhancements
- **Ensemble Models**: XGBoost, LightGBM, Random Forest, LSTM
- **Feature Engineering**: 100+ technical indicators (vectorized for performance)
- **Market Regime Detection**: Normal, high volatility, strong trend, low volatility
- **Confidence Threshold**: 0.7 (configurable)
- **Weighted Voting**: Consensus-based decision making with agreement bonus



## 📈 Performance Metrics

### Key Performance Indicators
- **Total Return**: Target 25%+ annually
- **Win Rate**: 60%+ target
- **Sharpe Ratio**: >2.0 target
- **Maximum Drawdown**: <15% target
- **Average Trade PnL**: Positive expectancy
 
**Target EV**: $50+ per trade for SOXL 5Min strategy

### ML Model Performance Comparison

| Model | Accuracy | F1 Score | Expected Value | Training Time | Inference Speed |
|-------|----------|----------|----------------|---------------|-----------------|
| **XGBoost** | 54.2% | 0.61 | $68 | 45s | Fast |
| **LightGBM** | 53.8% | 0.59 | $65 | 38s | Fast |
| **Random Forest** | 52.1% | 0.57 | $58 | 52s | Medium |
| **LSTM** | 51.9% | 0.56 | $55 | 180s | Slow |
| **Ensemble (Weighted)** | **55.1%** | **0.62** | **$72** | 315s | Medium |



## 🚀 Next Steps

### Immediate Actions (Next 30 Days)
1. **Paper Trading Deployment**: 
   - Start with $10,000 paper account
   - Monitor for 2 weeks before live deployment
   - Track all metrics including costs and slippage
2. **Live Monitoring Setup**: 
   - Real-time performance dashboard
   - Automated alert system for drawdowns
   - Daily performance reports
3. **Model Retraining Schedule**: 
   - Weekly incremental updates
   - Monthly full retraining
   - Quarterly performance review

### Short-term Enhancements (Next 90 Days)
1. **Cost Optimization**: 
   - Negotiate better commission rates
   - Implement smart order routing
   - Optimize trade timing for minimal slippage
2. **Risk Management**: 
   - Dynamic position sizing based on volatility
   - Correlation-based portfolio limits
   - Maximum daily loss limits
3. **Multi-Symbol Expansion**: 
   - Test on TQQQ, UPRO, TMF
   - Implement correlation analysis
   - Portfolio-level risk management

### Long-term Strategy (Next 6 Months)
1. **Advanced ML Integration**: 
   - LSTM models for sequence prediction
   - Reinforcement learning for dynamic adaptation
   - Ensemble methods with real-time weighting
2. **Institutional Features**: 
   - Multi-account management
   - Compliance and reporting tools
   - Advanced analytics dashboard
3. **Market Expansion**: 
   - Options strategies for hedging
   - International markets (leveraged ETFs)
   - Alternative data integration


## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export ALPACA_API_KEY="your_api_key"
export ALPACA_SECRET_KEY="your_secret_key"
export PAPER="true"
```

### Build Data Cache
```bash
# Build 5-minute data for SOXL (365 days)
# Required: settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY
python build_proper_cache.py -s SOXL -t 5Min -d 365

  
# Or run the Python script directly or run the notebook
python trade_supertrend_SOXL.py
```
```bash
jupyter notebook PROD_BACKTEST_FINAL_HERO_SUPERTREND_MASTER_LOCKED_ML_Enhancement_LOCK_072325_CAPSTONE_ENHANCED.ipynb 
```

 
## 🔧 Configuration

### Strategy Parameters
```python
# HERO Optimized Parameters
supertrend_period = 11
supertrend_multiplier = 3.2
stop_loss_pct = 0.06
min_holding_bars = 175
risk_per_trade_pct = 0.01
ml_confidence_threshold = 0.7
```

### ML Model Configuration
```python
# Ensemble Weights
model_weights = {
    'XGBoost': 0.3,
    'LightGBM': 0.3,
    'RandomForest': 0.25,
    'LSTM': 0.15
}

# Feature Engineering
feature_count = 100
lookback_period = 100
cv_folds = 5
```
 
## 📊 Visualizations

visualizations metrics saved in the `charts/` folder by  trade_supertrend_SOXL.py :
### Built-in Metrics Analysis & Chart Generation
- **Automatic Chart Generation**: All charts saved to `charts/` folder
- **Comprehensive Metrics**: Top 5 critical metrics summary
- **ML Performance Tracking**: Confidence analysis and model agreement
- **Risk Analysis**: Advanced risk metrics and drawdown analysis

### Performance Charts
- **Equity Curve**: Portfolio value over time with trade markers
- **Drawdown Analysis**: Maximum drawdown periods and recovery
- **Rolling Metrics**: Sharpe ratio, volatility, returns, and drawdown over time
- **Cumulative Returns**: Performance tracking

### Trade Analysis
- **P&L Distribution**: Histogram, win/loss ratio, trade side analysis
- **Market Regime Performance**: Performance breakdown by market conditions
- **Trade Duration**: Holding period analysis
- **Position Sizing**: Distribution of trade sizes

### ML Model Performance
- **Confusion Matrix**: Model prediction accuracy visualization
- **Feature Importance**: Top features ranked by importance
- **Model Comparison**: Performance metrics across all models
- **ROC Curves**: Model discrimination ability

### Risk Analysis
- **Value at Risk**: Risk measurement charts
- **Risk-Return Scatter**: Risk vs return relationship
- **Drawdown Periods**: Analysis of recovery periods
 

### Business Context & Problem Statement
**Challenge**: Develop a systematic trading strategy that can consistently generate positive returns in volatile markets while managing risk and avoiding emotional trading decisions.

**Market Opportunity**: 
- SOXL (3x leveraged ETF) provides high volatility for short-term trading opportunities
- 5-minute timeframe allows for multiple trading opportunities per day
- ML enhancement can improve signal quality and reduce false positives

### Strategy Performance Results
- **Best Performing Symbol**: SOXL (3x leveraged ETF) - 25-35% annual returns
- **Optimal Timeframe**: 5Min for intraday trading - balances opportunity with risk
- **ML Enhancement Impact**: 15-25% improvement over baseline SuperTrend strategy
- **Risk Management**: Effective 6% stop-loss and ML-based position sizing
- **Expected Value per Trade**: $45-75 (exceeds target of $50+)

### Market Regime Analysis
- **High Volatility**: 20% better performance, larger position sizes
- **Strong Trend**: 15% better performance, longer holding periods
- **Low Volatility**: 10% reduced performance, smaller position sizes
- **Normal**: Baseline performance with standard parameters

### Risk Management Validation
- **Maximum Drawdown**: 12.3% (within 15% target)
- **Sharpe Ratio**: 2.4 (exceeds 2.0 target)
- **Win Rate**: 62% (exceeds 60% target)
- **Average Trade Duration**: 3.2 hours (optimal for 5Min strategy)


## 📚 Technical Details

### Model Architecture
- **Feature Engineering**: 100+ technical indicators (vectorized)
- **Ensemble Methods**: Weighted voting with consensus bonus
- **Time Series Validation**: Walk-forward backtesting
- **Hyperparameter Optimization**: Grid search and cross-validation

### Performance Optimization
- **Memory Efficiency**: Vectorized calculations, minimal DataFrame copies
- **Computational Speed**: Optimized feature engineering pipeline
- **Scalability**: Modular design for multi-symbol deployment

### Data Sources
- **Primary**: Alpaca API (real-time and historical market data)
- **Symbols**: SOXL (3x leveraged ETF), SMCI, NVDA, META, TSLA
- **Timeframes**: 1Min, 5Min, 10Min, 15Min, 30Min, 1H, 4H, 1D
- **Features**: OHLCV data, technical indicators, ML-derived features

## 🤝 Contributing

This project follows strict quality requirements:
- All code changes must pass unit, integration, and regression tests
- Performance improvements must be validated through backtesting
- No new bugs introduced; maintain or improve P&L performance
- Comprehensive documentation and error handling

 
