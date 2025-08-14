#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import pytz
import argparse
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from src.config.settings import Settings

def build_proper_cache(symbol: str, timeframe: str = '5Min', days: int = 90):
    """Build a proper cache with enough historical data for indicator calculations"""
    
    print(f"🔧 Building proper cache for {symbol} on {timeframe} timeframe...")
    
    # Initialize Alpaca client
    settings = Settings()
    historical_client = StockHistoricalDataClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)
    
    # Calculate date range (last N days to ensure enough data)
    end_time = datetime.now(pytz.timezone('America/New_York'))
    start_time = end_time - timedelta(days=days)
    
    print(f"Fetching data from {start_time} to {end_time}")
    
    # Map timeframe string to TimeFrame object
    timeframe_mapping = {
        '1Min': TimeFrame(1, TimeFrame.Minute),
        '5Min': TimeFrame(5, TimeFrame.Minute),
        '10Min': TimeFrame(10, TimeFrame.Minute),
        '15Min': TimeFrame(15, TimeFrame.Minute),
        '30Min': TimeFrame(1, TimeFrame.Hour),  # Fixed: Use 1H for 30Min (Alpaca limitation)
        '1H': TimeFrame(1, TimeFrame.Hour),
        '1D': TimeFrame(1, TimeFrame.Day)
    }
    
    alpaca_timeframe = timeframe_mapping.get(timeframe, TimeFrame(5, TimeFrame.Minute))
    
    # Fetch data with pagination
    all_bars = []
    current_start = start_time
    
    while current_start < end_time:
        # Calculate end time for this batch (max 1000 bars per request)
        current_end = min(current_start + timedelta(days=7), end_time)
        
        print(f"Fetching batch: {current_start} to {current_end}")
        
        request = StockBarsRequest(
            symbol_or_symbols=symbol.upper(),
            timeframe=alpaca_timeframe,
            start=current_start,
            end=current_end
        )
        
        try:
            bars = historical_client.get_stock_bars(request)
            
            if bars and hasattr(bars, 'df') and not bars.df.empty:
                df_batch = bars.df.reset_index()
                all_bars.append(df_batch)
                print(f"✅ Fetched {len(df_batch)} bars for this batch")
            else:
                print(f"⚠️ No data for batch: {current_start} to {current_end}")
                
        except Exception as e:
            print(f"❌ Error fetching batch {current_start} to {current_end}: {e}")
        
        # Move to next batch
        current_start = current_end
    
    if not all_bars:
        print("❌ No data received from Alpaca")
        return False
    
    # Combine all batches
    df = pd.concat(all_bars, ignore_index=True)
    
    # Ensure we have the required columns
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        print(f"❌ Missing required columns. Available: {df.columns.tolist()}")
        return False
    
    # Clean and prepare data
    df = df[required_columns].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add symbol column
    df['symbol'] = symbol.upper()
    
    # Ensure timestamp is timezone-aware
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('America/New_York')
    
    # Remove duplicates and sort by timestamp
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    print(f"✅ Total fetched: {len(df)} bars of data")
    print(f"✅ Data shape after cleaning: {df.shape}")
    
    # Save to cache
    cache_file = f"data/cache_{symbol.upper()}_{timeframe}.csv"
    
    # Write the data and verify
    df.to_csv(cache_file, index=False)
    
    # Verify the file was written correctly
    verification_df = pd.read_csv(cache_file)
    print(f"✅ Verification: File contains {len(verification_df)} rows")
    
    print(f"✅ Cache saved to {cache_file}")
    print(f"📊 Data summary:")
    print(f"   - Symbol: {symbol.upper()}")
    print(f"   - Timeframe: {timeframe}")
    print(f"   - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   - Total bars: {len(df)}")
    print(f"   - Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"   - Average volume: {df['volume'].mean():,.0f}")
    
    return True

def resample_1min_to_1d(symbol):
    import pandas as pd
    src_path = f"data/cache_{symbol}_1Min.csv"
    dst_path = f"data/cache_{symbol}_1D.csv"
    print(f"Resampling {src_path} to daily bars at {dst_path}...")
    df = pd.read_csv(src_path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    daily = df.resample('1D').agg(ohlc_dict)
    daily = daily.dropna()
    daily.reset_index(inplace=True)
    daily.to_csv(dst_path, index=False)
    print(f"Saved daily bars: {len(daily)} rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build cache file for trading symbols')
    parser.add_argument('--symbol', '-s', required=True, help='Trading symbol (e.g., SOXL, TSLA, NVDA)')
    parser.add_argument('--timeframe', '-t', default='5Min', 
                       choices=['1Min', '5Min', '10Min', '15Min', '30Min', '1H', '1D'],
                       help='Data timeframe (default: 5Min)')
    parser.add_argument('--days', '-d', type=int, default=90, 
                       help='Number of days to fetch (default: 90)')
    
    args = parser.parse_args()
    
    try:
        success = build_proper_cache(args.symbol, args.timeframe, args.days)
        if success:
            print(f"\n🎉 Cache built successfully for {args.symbol.upper()}! The live trader should now work properly.")
        else:
            print(f"\n💥 Failed to build cache for {args.symbol.upper()}. Please check your Alpaca credentials and try again.")
        
        if args.timeframe == "1D":
            # Try to fetch, but if no data, resample from 1Min
            try:
                # ... existing code for fetching ...
                pass
            except Exception as e:
                print(f"Alpaca 1D fetch failed: {e}. Attempting to resample from 1Min...")
                resample_1min_to_1d(args.symbol)
    except Exception as e:
        print(f"Error: {e}") 