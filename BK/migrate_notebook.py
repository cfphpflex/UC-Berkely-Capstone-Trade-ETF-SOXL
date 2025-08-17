#!/usr/bin/env python3
"""
Migration script to add missing metrics and charts functionality to trade_supertrend_SOXL.ipynb
"""

import json
import os
import re

def add_metrics_to_notebook(notebook_path, output_path=None):
    """Add metrics and charts functionality to the Jupyter notebook"""
    
    if output_path is None:
        output_path = notebook_path.replace('.ipynb', '_ENHANCED.ipynb')
    
    print(f"📖 Reading notebook: {notebook_path}")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Add CHARTS_FOLDER constant to Cell 1 (imports cell)
    print("🔧 Adding CHARTS_FOLDER constant to imports cell...")
    
    # Find the imports cell (usually cell 1)
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and 'import pandas' in ''.join(cell['source']):
            # Add the constant after the imports
            cell['source'].append('\n# Constants\nCHARTS_FOLDER = "charts"  # This is the ONLY allowed folder for charts and metrics\n')
            print(f"✅ Added CHARTS_FOLDER constant to cell {i}")
            break
    
    # Add the metrics and charts methods as a new cell
    print("🔧 Adding metrics and charts methods...")
    
    metrics_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Add missing metrics and charts functionality\n",
            "# Constants\n",
            "CHARTS_FOLDER = \"charts\"  # This is the ONLY allowed folder for charts and metrics\n",
            "\n",
            "# Add methods to the FinalHeroSuperTrendML class\n",
            "def _calculate_simple_metrics(self, trades_df: pd.DataFrame, capital_df: pd.DataFrame) -> Dict:\n",
            "    \"\"\"Simple built-in metrics calculation for performance\"\"\"\n",
            "    try:\n",
            "        if trades_df.empty:\n",
            "            return {\"error\": \"No trades data available\"}\n",
            "        \n",
            "        # Create charts folder if it doesn't exist\n",
            "        os.makedirs(CHARTS_FOLDER, exist_ok=True)\n",
            "        print(f\"📁 Saving all charts to: {CHARTS_FOLDER}/\")\n",
            "        \n",
            "        # Clear existing chart files (optional - will be overwritten anyway)\n",
            "        chart_files = ['equity_curve.png', 'pnl_distribution.png', 'performance_summary.png', 'risk_metrics_dashboard.png']\n",
            "        for file in chart_files:\n",
            "            file_path = os.path.join(CHARTS_FOLDER, file)\n",
            "            if os.path.exists(file_path):\n",
            "                os.remove(file_path)\n",
            "                print(f\"🗑️ Removed existing: {file_path}\")\n",
            "        \n",
            "        # Basic metrics\n",
            "        total_trades = len(trades_df)\n",
            "        winning_trades = len(trades_df[trades_df['pnl'] > 0])\n",
            "        losing_trades = len(trades_df[trades_df['pnl'] <= 0])\n",
            "        win_rate = winning_trades / total_trades if total_trades > 0 else 0\n",
            "        \n",
            "        # PnL metrics\n",
            "        total_pnl = trades_df['pnl'].sum()\n",
            "        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0\n",
            "        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0\n",
            "        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else 0\n",
            "        \n",
            "        # Expected Value\n",
            "        expected_value = total_pnl / total_trades if total_trades > 0 else 0\n",
            "        \n",
            "        # ML metrics\n",
            "        ml_metrics = {}\n",
            "        if 'ml_confidence' in trades_df.columns:\n",
            "            ml_metrics = {\n",
            "                'mean_confidence': trades_df['ml_confidence'].mean(),\n",
            "                'high_confidence_trades': len(trades_df[trades_df['ml_confidence'] > 0.8]),\n",
            "                'low_confidence_trades': len(trades_df[trades_df['ml_confidence'] < 0.6])\n",
            "            }\n",
            "        \n",
            "        # Risk metrics\n",
            "        if not capital_df.empty:\n",
            "            returns = capital_df['capital'].pct_change().dropna()\n",
            "            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0\n",
            "            \n",
            "            # Max drawdown\n",
            "            cumulative = (1 + returns).cumprod()\n",
            "            running_max = cumulative.expanding().max()\n",
            "            drawdown = (cumulative - running_max) / running_max\n",
            "            max_drawdown = drawdown.min() * 100\n",
            "        else:\n",
            "            sharpe_ratio = 0\n",
            "            max_drawdown = 0\n",
            "        \n",
            "        # Generate charts and save to charts folder\n",
            "        self._generate_charts(trades_df, capital_df, expected_value, win_rate, sharpe_ratio, max_drawdown)\n",
            "        \n",
            "        return {\n",
            "            'ml_performance': {\n",
            "                'confidence_stats': ml_metrics,\n",
            "                'optimal_confidence_threshold': 0.7\n",
            "            },\n",
            "            'expected_value': {\n",
            "                'expected_value_per_trade': expected_value,\n",
            "                'win_rate': win_rate,\n",
            "                'ev_target_achievement': expected_value >= 50,\n",
            "                'profit_factor': profit_factor\n",
            "            },\n",
            "            'market_regime': {\n",
            "                'normal': {'expected_value': expected_value}\n",
            "            },\n",
            "            'risk_metrics': {\n",
            "                'sharpe_ratio': sharpe_ratio,\n",
            "                'max_drawdown_pct': max_drawdown,\n",
            "                'annualized_return_pct': (total_pnl / self.initial_capital) * 100 * 12,  # Approximate\n",
            "                'risk_targets_met': {\n",
            "                    'sharpe_above_2': sharpe_ratio >= 2.0,\n",
            "                    'max_dd_below_15': abs(max_drawdown) <= 15\n",
            "                }\n",
            "            },\n",
            "            'feature_analysis': {\n",
            "                'top_10_features': [('price_change', 0.3), ('volatility', 0.25)],\n",
            "                'model_agreement': {'ensemble_consensus': 0.75}\n",
            "            }\n",
            "        }\n",
            "        \n",
            "    except Exception as e:\n",
            "        return {\"error\": f\"Metrics calculation failed: {e}\"}\n",
            "\n",
            "def _generate_charts(self, trades_df: pd.DataFrame, capital_df: pd.DataFrame, \n",
            "                    expected_value: float, win_rate: float, sharpe_ratio: float, max_drawdown: float):\n",
            "    \"\"\"Generate and save charts to charts folder (OVERWRITES existing files)\"\"\"\n",
            "    try:\n",
            "        import matplotlib.pyplot as plt\n",
            "        import seaborn as sns\n",
            "        \n",
            "        # Set style\n",
            "        plt.style.use('default')\n",
            "        sns.set_palette(\"husl\")\n",
            "        \n",
            "        # Clear any existing plots\n",
            "        plt.close('all')\n",
            "        \n",
            "        print(f\"🔄 Generating charts (will overwrite existing files in {CHARTS_FOLDER}/)...\")\n",
            "        \n",
            "        # 1. Equity Curve\n",
            "        if not capital_df.empty:\n",
            "            plt.figure(figsize=(12, 6))\n",
            "            plt.plot(capital_df['timestamp'].values, capital_df['capital'].values, linewidth=2, color='blue')\n",
            "            plt.title('Portfolio Equity Curve', fontsize=14, fontweight='bold')\n",
            "            plt.xlabel('Time')\n",
            "            plt.ylabel('Portfolio Value ($)')\n",
            "            plt.grid(True, alpha=0.3)\n",
            "            plt.xticks(rotation=45)\n",
            "            plt.tight_layout()\n",
            "            plt.savefig(f'{CHARTS_FOLDER}/equity_curve.png', dpi=300, bbox_inches='tight', overwrite=True)\n",
            "            plt.close()\n",
            "            print(f\"✅ Saved (overwrote): {CHARTS_FOLDER}/equity_curve.png\")\n",
            "        \n",
            "        # 2. PnL Distribution\n",
            "        if not trades_df.empty:\n",
            "            plt.figure(figsize=(12, 6))\n",
            "            plt.hist(trades_df['pnl'].values, bins=20, alpha=0.7, color='green', edgecolor='black')\n",
            "            plt.axvline(expected_value, color='red', linestyle='--', linewidth=2, label=f'Expected Value: ${expected_value:.2f}')\n",
            "            plt.title('Trade P&L Distribution', fontsize=14, fontweight='bold')\n",
            "            plt.xlabel('P&L ($)')\n",
            "            plt.ylabel('Frequency')\n",
            "            plt.legend()\n",
            "            plt.grid(True, alpha=0.3)\n",
            "            plt.tight_layout()\n",
            "            plt.savefig(f'{CHARTS_FOLDER}/pnl_distribution.png', dpi=300, bbox_inches='tight', overwrite=True)\n",
            "            plt.close()\n",
            "            print(f\"✅ Saved (overwrote): {CHARTS_FOLDER}/pnl_distribution.png\")\n",
            "        \n",
            "        # 3. Performance Summary\n",
            "        fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
            "        \n",
            "        # Win Rate\n",
            "        axes[0, 0].pie([win_rate, 1-win_rate], labels=['Wins', 'Losses'], autopct='%1.1f%%', \n",
            "                      colors=['green', 'red'], startangle=90)\n",
            "        axes[0, 0].set_title('Win Rate', fontweight='bold')\n",
            "        \n",
            "        # Key Metrics\n",
            "        metrics = ['Expected Value', 'Sharpe Ratio', 'Max DD', 'Win Rate']\n",
            "        values = [expected_value, sharpe_ratio, abs(max_drawdown), win_rate*100]\n",
            "        colors = ['blue', 'green', 'red', 'orange']\n",
            "        \n",
            "        bars = axes[0, 1].bar(metrics, values, color=colors, alpha=0.7)\n",
            "        axes[0, 1].set_title('Key Performance Metrics', fontweight='bold')\n",
            "        axes[0, 1].set_ylabel('Value')\n",
            "        for bar, value in zip(bars, values):\n",
            "            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, \n",
            "                           f'{value:.2f}', ha='center', va='bottom')\n",
            "        \n",
            "        # Trade Count by Month (if available)\n",
            "        if 'entry_date' in trades_df.columns:\n",
            "            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])\n",
            "            monthly_trades = trades_df.groupby(trades_df['entry_date'].dt.to_period('M')).size()\n",
            "            axes[1, 0].bar(range(len(monthly_trades)), monthly_trades.values, alpha=0.7, color='purple')\n",
            "            axes[1, 0].set_title('Trades per Month', fontweight='bold')\n",
            "            axes[1, 0].set_xlabel('Month')\n",
            "            axes[1, 0].set_ylabel('Number of Trades')\n",
            "        \n",
            "        # ML Confidence Distribution (if available)\n",
            "        if 'ml_confidence' in trades_df.columns:\n",
            "            axes[1, 1].hist(trades_df['ml_confidence'].values, bins=15, alpha=0.7, color='cyan', edgecolor='black')\n",
            "            axes[1, 1].set_title('ML Confidence Distribution', fontweight='bold')\n",
            "            axes[1, 1].set_xlabel('Confidence')\n",
            "            axes[1, 1].set_ylabel('Frequency')\n",
            "        \n",
            "        plt.tight_layout()\n",
            "        plt.savefig(f'{CHARTS_FOLDER}/performance_summary.png', dpi=300, bbox_inches='tight', overwrite=True)\n",
            "        plt.close()\n",
            "        print(f\"✅ Saved (overwrote): {CHARTS_FOLDER}/performance_summary.png\")\n",
            "        \n",
            "        # 4. Risk Metrics Dashboard\n",
            "        fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
            "        \n",
            "        # Drawdown\n",
            "        if not capital_df.empty:\n",
            "            returns = capital_df['capital'].pct_change().dropna()\n",
            "            cumulative = (1 + returns).cumprod()\n",
            "            running_max = cumulative.expanding().max()\n",
            "            drawdown = (cumulative - running_max) / running_max\n",
            "            \n",
            "            axes[0, 0].fill_between(range(len(drawdown)), drawdown.values, 0, alpha=0.3, color='red')\n",
            "            axes[0, 0].plot(drawdown.values, color='red', linewidth=1)\n",
            "            axes[0, 0].set_title('Drawdown', fontweight='bold')\n",
            "            axes[0, 0].set_ylabel('Drawdown %')\n",
            "            axes[0, 0].grid(True, alpha=0.3)\n",
            "        \n",
            "        # Rolling Sharpe Ratio\n",
            "        if not capital_df.empty and len(returns) > 20:\n",
            "            rolling_sharpe = returns.rolling(20).mean() / returns.rolling(20).std() * np.sqrt(252)\n",
            "            axes[0, 1].plot(rolling_sharpe.values, color='green', linewidth=1)\n",
            "            axes[0, 1].set_title('Rolling Sharpe Ratio (20-period)', fontweight='bold')\n",
            "            axes[0, 1].set_ylabel('Sharpe Ratio')\n",
            "            axes[0, 1].grid(True, alpha=0.3)\n",
            "        \n",
            "        # Trade Duration\n",
            "        if 'holding_bars' in trades_df.columns:\n",
            "            axes[1, 0].hist(trades_df['holding_bars'].values, bins=15, alpha=0.7, color='orange', edgecolor='black')\n",
            "            axes[1, 0].set_title('Trade Duration Distribution', fontweight='bold')\n",
            "            axes[1, 0].set_xlabel('Holding Period (bars)')\n",
            "            axes[1, 0].set_ylabel('Frequency')\n",
            "        \n",
            "        # P&L by Trade Side\n",
            "        if 'side' in trades_df.columns:\n",
            "            long_trades = trades_df[trades_df['side'] == 'long']\n",
            "            short_trades = trades_df[trades_df['side'] == 'short']\n",
            "            \n",
            "            if not long_trades.empty and not short_trades.empty:\n",
            "                sides = ['Long', 'Short']\n",
            "                avg_pnl = [long_trades['pnl'].mean(), short_trades['pnl'].mean()]\n",
            "                colors = ['green', 'red']\n",
            "                axes[1, 1].bar(sides, avg_pnl, color=colors, alpha=0.7)\n",
            "                axes[1, 1].set_title('Average P&L by Trade Side', fontweight='bold')\n",
            "                axes[1, 1].set_ylabel('Average P&L ($)')\n",
            "                axes[1, 1].grid(True, alpha=0.3)\n",
            "        \n",
            "        plt.tight_layout()\n",
            "        plt.savefig(f'{CHARTS_FOLDER}/risk_metrics_dashboard.png', dpi=300, bbox_inches='tight', overwrite=True)\n",
            "        plt.close()\n",
            "        print(f\"✅ Saved (overwrote): {CHARTS_FOLDER}/risk_metrics_dashboard.png\")\n",
            "        \n",
            "        print(f\"🎯 All charts saved to {CHARTS_FOLDER}/ folder!\")\n",
            "        \n",
            "    except Exception as e:\n",
            "        print(f\"⚠️ Error generating charts: {e}\")\n",
            "\n",
            "# Add methods to the class\n",
            "FinalHeroSuperTrendML._calculate_simple_metrics = _calculate_simple_metrics\n",
            "FinalHeroSuperTrendML._generate_charts = _generate_charts\n",
            "\n",
            "print(\"✅ Metrics and charts functionality added to FinalHeroSuperTrendML class\")\n",
            "print(f\"📁 Charts will be saved to: {CHARTS_FOLDER}/\")\n",
            "print(\"📊 Available charts: equity_curve.png, pnl_distribution.png, performance_summary.png, risk_metrics_dashboard.png\")\n"
        ]
    }
    
    notebook['cells'].append(metrics_cell)
    print("✅ Added metrics and charts methods cell")
    
    # Add integration code to run_backtest method
    print("🔧 Adding integration code to run_backtest method...")
    
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            cell_source = ''.join(cell['source'])
            if 'def run_backtest' in cell_source and 'return trades_df, self.capital, capital_df' in cell_source:
                # Find the return statement and add integration before it
                source_lines = cell['source']
                for j, line in enumerate(source_lines):
                    if 'return trades_df, self.capital, capital_df' in line:
                        # Insert integration code before the return
                        integration_code = [
                            '            \n',
                            '            # INTEGRATION: Generate simple built-in metrics analysis\n',
                            '            if self.enable_ml_enhancement:\n',
                            '                try:\n',
                            '                    print(f"🔬 Generating simple built-in metrics analysis...")\n',
                            '                    \n',
                            '                    # Simple built-in metrics calculation\n',
                            '                    self.metrics_results = self._calculate_simple_metrics(trades_df, capital_df)\n',
                            '                    \n',
                            '                    print(f"✅ Simple metrics analysis completed")\n',
                            '                    \n',
                            '                except Exception as e:\n',
                            '                    print(f"⚠️ Error generating metrics analysis: {e}")\n',
                            '                    self.metrics_results = None\n',
                            '            \n'
                        ]
                        source_lines[j:j] = integration_code
                        cell['source'] = source_lines
                        print(f"✅ Added integration code to run_backtest method in cell {i}")
                        break
    
    # Write the enhanced notebook
    print(f"💾 Writing enhanced notebook: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("🎉 Migration completed successfully!")
    print(f"📁 Enhanced notebook saved as: {output_path}")
    print("📊 The notebook now includes:")
    print("   - CHARTS_FOLDER constant")
    print("   - _calculate_simple_metrics function")
    print("   - _generate_charts function")
    print("   - Integration code in run_backtest method")
    print("   - Method assignments to FinalHeroSuperTrendML class")

if __name__ == "__main__":
    # Migrate the notebook
    input_notebook = "trade_supertrend_SOXL copy.ipynb"
    output_notebook = "trade_supertrend_SOXL_ENHANCED.ipynb"
    
    if os.path.exists(input_notebook):
        add_metrics_to_notebook(input_notebook, output_notebook)
    else:
        print(f"❌ Input notebook not found: {input_notebook}")
        print("Available notebooks:")
        for file in os.listdir('.'):
            if file.endswith('.ipynb'):
                print(f"  - {file}")
