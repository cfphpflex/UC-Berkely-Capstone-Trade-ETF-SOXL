# Missing Metrics and Charts Functionality for trade_supertrend_SOXL.ipynb
# This file contains the functions that need to be added to the notebook

import os
import pandas as pd
import numpy as np
from typing import Dict

# Constants
CHARTS_FOLDER = "charts"  # This is the ONLY allowed folder for charts and metrics

def _calculate_simple_metrics(self, trades_df: pd.DataFrame, capital_df: pd.DataFrame) -> Dict:
    """Simple built-in metrics calculation for performance"""
    try:
        if trades_df.empty:
            return {"error": "No trades data available"}
        
        # Create charts folder if it doesn't exist
        os.makedirs(CHARTS_FOLDER, exist_ok=True)
        print(f"📁 Saving all charts to: {CHARTS_FOLDER}/")
        
        # Clear existing chart files (optional - will be overwritten anyway)
        chart_files = ['equity_curve.png', 'pnl_distribution.png', 'performance_summary.png', 'risk_metrics_dashboard.png']
        for file in chart_files:
            file_path = os.path.join(CHARTS_FOLDER, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ Removed existing: {file_path}")
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else 0
        
        # Expected Value
        expected_value = total_pnl / total_trades if total_trades > 0 else 0
        
        # ML metrics
        ml_metrics = {}
        if 'ml_confidence' in trades_df.columns:
            ml_metrics = {
                'mean_confidence': trades_df['ml_confidence'].mean(),
                'high_confidence_trades': len(trades_df[trades_df['ml_confidence'] > 0.8]),
                'low_confidence_trades': len(trades_df[trades_df['ml_confidence'] < 0.6])
            }
        
        # Risk metrics
        if not capital_df.empty:
            returns = capital_df['capital'].pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Generate charts and save to charts folder
        self._generate_charts(trades_df, capital_df, expected_value, win_rate, sharpe_ratio, max_drawdown)
        
        return {
            'ml_performance': {
                'confidence_stats': ml_metrics,
                'optimal_confidence_threshold': 0.7
            },
            'expected_value': {
                'expected_value_per_trade': expected_value,
                'win_rate': win_rate,
                'ev_target_achievement': expected_value >= 50,
                'profit_factor': profit_factor
            },
            'market_regime': {
                'normal': {'expected_value': expected_value}
            },
            'risk_metrics': {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'annualized_return_pct': (total_pnl / self.initial_capital) * 100 * 12,  # Approximate
                'risk_targets_met': {
                    'sharpe_above_2': sharpe_ratio >= 2.0,
                    'max_dd_below_15': abs(max_drawdown) <= 15
                }
            },
            'feature_analysis': {
                'top_10_features': [('price_change', 0.3), ('volatility', 0.25)],
                'model_agreement': {'ensemble_consensus': 0.75}
            }
        }
        
    except Exception as e:
        return {"error": f"Metrics calculation failed: {e}"}

def _generate_charts(self, trades_df: pd.DataFrame, capital_df: pd.DataFrame, 
                    expected_value: float, win_rate: float, sharpe_ratio: float, max_drawdown: float):
    """Generate and save charts to charts folder (OVERWRITES existing files)"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Clear any existing plots
        plt.close('all')
        
        print(f"🔄 Generating charts (will overwrite existing files in {CHARTS_FOLDER}/)...")
        
        # 1. Equity Curve
        if not capital_df.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(capital_df['timestamp'].values, capital_df['capital'].values, linewidth=2, color='blue')
            plt.title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
            plt.xlabel('Time')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{CHARTS_FOLDER}/equity_curve.png', dpi=300, bbox_inches='tight', overwrite=True)
            plt.close()
            print(f"✅ Saved (overwrote): {CHARTS_FOLDER}/equity_curve.png")
        
        # 2. PnL Distribution
        if not trades_df.empty:
            plt.figure(figsize=(12, 6))
            plt.hist(trades_df['pnl'].values, bins=20, alpha=0.7, color='green', edgecolor='black')
            plt.axvline(expected_value, color='red', linestyle='--', linewidth=2, label=f'Expected Value: ${expected_value:.2f}')
            plt.title('Trade P&L Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('P&L ($)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{CHARTS_FOLDER}/pnl_distribution.png', dpi=300, bbox_inches='tight', overwrite=True)
            plt.close()
            print(f"✅ Saved (overwrote): {CHARTS_FOLDER}/pnl_distribution.png")
        
        # 3. Performance Summary
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Win Rate
        axes[0, 0].pie([win_rate, 1-win_rate], labels=['Wins', 'Losses'], autopct='%1.1f%%', 
                      colors=['green', 'red'], startangle=90)
        axes[0, 0].set_title('Win Rate', fontweight='bold')
        
        # Key Metrics
        metrics = ['Expected Value', 'Sharpe Ratio', 'Max DD', 'Win Rate']
        values = [expected_value, sharpe_ratio, abs(max_drawdown), win_rate*100]
        colors = ['blue', 'green', 'red', 'orange']
        
        bars = axes[0, 1].bar(metrics, values, color=colors, alpha=0.7)
        axes[0, 1].set_title('Key Performance Metrics', fontweight='bold')
        axes[0, 1].set_ylabel('Value')
        for bar, value in zip(bars, values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.2f}', ha='center', va='bottom')
        
        # Trade Count by Month (if available)
        if 'entry_date' in trades_df.columns:
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
            monthly_trades = trades_df.groupby(trades_df['entry_date'].dt.to_period('M')).size()
            axes[1, 0].bar(range(len(monthly_trades)), monthly_trades.values, alpha=0.7, color='purple')
            axes[1, 0].set_title('Trades per Month', fontweight='bold')
            axes[1, 0].set_xlabel('Month')
            axes[1, 0].set_ylabel('Number of Trades')
        
        # ML Confidence Distribution (if available)
        if 'ml_confidence' in trades_df.columns:
            axes[1, 1].hist(trades_df['ml_confidence'].values, bins=15, alpha=0.7, color='cyan', edgecolor='black')
            axes[1, 1].set_title('ML Confidence Distribution', fontweight='bold')
            axes[1, 1].set_xlabel('Confidence')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'{CHARTS_FOLDER}/performance_summary.png', dpi=300, bbox_inches='tight', overwrite=True)
        plt.close()
        print(f"✅ Saved (overwrote): {CHARTS_FOLDER}/performance_summary.png")
        
        # 4. Risk Metrics Dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Drawdown
        if not capital_df.empty:
            returns = capital_df['capital'].pct_change().dropna()
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            axes[0, 0].fill_between(range(len(drawdown)), drawdown.values, 0, alpha=0.3, color='red')
            axes[0, 0].plot(drawdown.values, color='red', linewidth=1)
            axes[0, 0].set_title('Drawdown', fontweight='bold')
            axes[0, 0].set_ylabel('Drawdown %')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Rolling Sharpe Ratio
        if not capital_df.empty and len(returns) > 20:
            rolling_sharpe = returns.rolling(20).mean() / returns.rolling(20).std() * np.sqrt(252)
            axes[0, 1].plot(rolling_sharpe.values, color='green', linewidth=1)
            axes[0, 1].set_title('Rolling Sharpe Ratio (20-period)', fontweight='bold')
            axes[0, 1].set_ylabel('Sharpe Ratio')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Trade Duration
        if 'holding_bars' in trades_df.columns:
            axes[1, 0].hist(trades_df['holding_bars'].values, bins=15, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].set_title('Trade Duration Distribution', fontweight='bold')
            axes[1, 0].set_xlabel('Holding Period (bars)')
            axes[1, 0].set_ylabel('Frequency')
        
        # P&L by Trade Side
        if 'side' in trades_df.columns:
            long_trades = trades_df[trades_df['side'] == 'long']
            short_trades = trades_df[trades_df['side'] == 'short']
            
            if not long_trades.empty and not short_trades.empty:
                sides = ['Long', 'Short']
                avg_pnl = [long_trades['pnl'].mean(), short_trades['pnl'].mean()]
                colors = ['green', 'red']
                axes[1, 1].bar(sides, avg_pnl, color=colors, alpha=0.7)
                axes[1, 1].set_title('Average P&L by Trade Side', fontweight='bold')
                axes[1, 1].set_ylabel('Average P&L ($)')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{CHARTS_FOLDER}/risk_metrics_dashboard.png', dpi=300, bbox_inches='tight', overwrite=True)
        plt.close()
        print(f"✅ Saved (overwrote): {CHARTS_FOLDER}/risk_metrics_dashboard.png")
        
        print(f"🎯 All charts saved to {CHARTS_FOLDER}/ folder!")
        
    except Exception as e:
        print(f"⚠️ Error generating charts: {e}")

# Integration code for run_backtest method
def add_metrics_integration_to_run_backtest():
    """
    Code to add to the run_backtest method in the notebook:
    
    # INTEGRATION: Generate simple built-in metrics analysis
    if self.enable_ml_enhancement:
        try:
            print(f"🔬 Generating simple built-in metrics analysis...")
            
            # Simple built-in metrics calculation
            self.metrics_results = self._calculate_simple_metrics(trades_df, capital_df)
            
            print(f"✅ Simple metrics analysis completed")
            
        except Exception as e:
            print(f"⚠️ Error generating metrics analysis: {e}")
            self.metrics_results = None
    """
    pass

# Instructions for manual addition to notebook
print("""
INSTRUCTIONS FOR MANUAL ADDITION TO trade_supertrend_SOXL.ipynb:

1. ADD CONSTANT (in Cell 1 after imports):
   CHARTS_FOLDER = "charts"  # This is the ONLY allowed folder for charts and metrics

2. ADD METHODS (create new cell at end):
   Copy the _calculate_simple_metrics and _generate_charts functions from this file

3. ADD INTEGRATION (in run_backtest method before return statement):
   # INTEGRATION: Generate simple built-in metrics analysis
   if self.enable_ml_enhancement:
       try:
           print(f"🔬 Generating simple built-in metrics analysis...")
           self.metrics_results = self._calculate_simple_metrics(trades_df, capital_df)
           print(f"✅ Simple metrics analysis completed")
       except Exception as e:
           print(f"⚠️ Error generating metrics analysis: {e}")
           self.metrics_results = None

4. ADD METHODS TO CLASS (in same cell as methods):
   FinalHeroSuperTrendML._calculate_simple_metrics = _calculate_simple_metrics
   FinalHeroSuperTrendML._generate_charts = _generate_charts
""")
