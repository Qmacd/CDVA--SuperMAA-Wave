
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Calculate various technical indicators for financial data"""
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: Price series (typically close prices)
            period: RSI period (default 14)
            
        Returns:
            RSI values (0-100)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral value
    
    @staticmethod
    def calculate_momentum(prices, period=10):
        """
        Calculate Price Momentum
        
        Args:
            prices: Price series
            period: Momentum period (default 10)
            
        Returns:
            Momentum values
        """
        momentum = prices.pct_change(periods=period) * 100
        return momentum.fillna(0)
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Price series
            period: Moving average period (default 20)
            std_dev: Standard deviation multiplier (default 2)
            
        Returns:
            Dictionary with upper_band, middle_band, lower_band, bandwidth, %b
        """
        middle_band = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        bandwidth = (upper_band - lower_band) / middle_band * 100
        
        percent_b = (prices - lower_band) / (upper_band - lower_band) * 100
        
        return {
            'upper_band': upper_band.fillna(prices),
            'middle_band': middle_band.fillna(prices),
            'lower_band': lower_band.fillna(prices),
            'bandwidth': bandwidth.fillna(20),  # Default bandwidth
            'percent_b': percent_b.fillna(50)   # Default middle position
        }
    
    @staticmethod
    def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Price series
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line EMA period (default 9)
            
        Returns:
            Dictionary with macd_line, signal_line, histogram
        """
        ema_fast = prices.ewm(span=fast_period).mean()
        ema_slow = prices.ewm(span=slow_period).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd_line': macd_line.fillna(0),
            'signal_line': signal_line.fillna(0),
            'histogram': histogram.fillna(0)
        }
    
    @staticmethod
    def calculate_stochastic(high, low, close, k_period=14, d_period=3):
        """
        Calculate Stochastic Oscillator
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: %K period (default 14)
            d_period: %D period (default 3)
            
        Returns:
            Dictionary with %K and %D values
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k_percent': k_percent.fillna(50),
            'd_percent': d_percent.fillna(50)
        }
    
    @staticmethod
    def calculate_williams_r(high, low, close, period=14):
        """
        Calculate Williams %R
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Period (default 14)
            
        Returns:
            Williams %R values
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
        return williams_r.fillna(-50)
    
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        """
        Calculate Average True Range (ATR)
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period (default 14)
            
        Returns:
            ATR values
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr.fillna(true_range.mean())
    
    @staticmethod
    def calculate_cci(high, low, close, period=20):
        """
        Calculate Commodity Channel Index (CCI)
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: CCI period (default 20)
            
        Returns:
            CCI values
        """
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        return cci.fillna(0)
    
    @classmethod
    def calculate_all_indicators(cls, df):
        """
        Calculate all technical indicators for a dataframe
        
        Args:
            df: DataFrame with OHLC data (open, high, low, close)
            
        Returns:
            DataFrame with all technical indicators added
        """
        result_df = df.copy()
        
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in result_df.columns:
                print(f"⚠️  Missing column: {col}, using close price as substitute")
                result_df[col] = result_df.get('close', result_df.iloc[:, 0])
        
        result_df['rsi'] = cls.calculate_rsi(result_df['close'])
        
        result_df['momentum'] = cls.calculate_momentum(result_df['close'])
        
        bb = cls.calculate_bollinger_bands(result_df['close'])
        result_df['bb_upper'] = bb['upper_band']
        result_df['bb_middle'] = bb['middle_band']
        result_df['bb_lower'] = bb['lower_band']
        result_df['bb_bandwidth'] = bb['bandwidth']
        result_df['bb_percent_b'] = bb['percent_b']
        
        macd = cls.calculate_macd(result_df['close'])
        result_df['macd_line'] = macd['macd_line']
        result_df['macd_signal'] = macd['signal_line']
        result_df['macd_histogram'] = macd['histogram']
        
        stoch = cls.calculate_stochastic(result_df['high'], result_df['low'], result_df['close'])
        result_df['stoch_k'] = stoch['k_percent']
        result_df['stoch_d'] = stoch['d_percent']
        
        result_df['williams_r'] = cls.calculate_williams_r(
            result_df['high'], result_df['low'], result_df['close']
        )
        
        result_df['atr'] = cls.calculate_atr(
            result_df['high'], result_df['low'], result_df['close']
        )
        
        result_df['cci'] = cls.calculate_cci(
            result_df['high'], result_df['low'], result_df['close']
        )
        
        return result_df
    
    @staticmethod
    def get_indicator_summary(df):
        """
        Get summary statistics for all technical indicators
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Dictionary with indicator summaries
        """
        indicator_cols = [
            'rsi', 'momentum', 'bb_bandwidth', 'bb_percent_b',
            'macd_line', 'macd_signal', 'macd_histogram',
            'stoch_k', 'stoch_d', 'williams_r', 'atr', 'cci'
        ]
        
        summary = {}
        for col in indicator_cols:
            if col in df.columns:
                summary[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'current': df[col].iloc[-1] if len(df) > 0 else 0
                }
        
        return summary
    
    @staticmethod
    def interpret_indicators(df):
        """
        Provide interpretation of technical indicators
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Dictionary with interpretations
        """
        if len(df) == 0:
            return {}
        
        latest = df.iloc[-1]
        interpretations = {}
        
        if 'rsi' in df.columns:
            rsi = latest['rsi']
            if rsi > 70:
                interpretations['rsi'] = 'Overbought (>70)'
            elif rsi < 30:
                interpretations['rsi'] = 'Oversold (<30)'
            else:
                interpretations['rsi'] = 'Neutral (30-70)'
        
        if 'bb_percent_b' in df.columns:
            bb_b = latest['bb_percent_b']
            if bb_b > 100:
                interpretations['bollinger'] = 'Above Upper Band (Overbought)'
            elif bb_b < 0:
                interpretations['bollinger'] = 'Below Lower Band (Oversold)'
            elif bb_b > 80:
                interpretations['bollinger'] = 'Near Upper Band'
            elif bb_b < 20:
                interpretations['bollinger'] = 'Near Lower Band'
            else:
                interpretations['bollinger'] = 'Within Bands'
        
        if 'macd_line' in df.columns and 'macd_signal' in df.columns:
            macd_line = latest['macd_line']
            macd_signal = latest['macd_signal']
            if macd_line > macd_signal:
                interpretations['macd'] = 'Bullish (MACD > Signal)'
            else:
                interpretations['macd'] = 'Bearish (MACD < Signal)'
        
        if 'stoch_k' in df.columns:
            stoch_k = latest['stoch_k']
            if stoch_k > 80:
                interpretations['stochastic'] = 'Overbought (>80)'
            elif stoch_k < 20:
                interpretations['stochastic'] = 'Oversold (<20)'
            else:
                interpretations['stochastic'] = 'Neutral (20-80)'
        
        return interpretations

def add_technical_indicators_to_results(results_df, sample_data=None):
    """
    Add technical indicators analysis to experiment results
    
    Args:
        results_df: Experiment results DataFrame
        sample_data: Sample price data for indicator calculation
        
    Returns:
        Enhanced results DataFrame with technical indicators
    """
    if sample_data is None:
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'date': dates,
            'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'high': 0,
            'low': 0,
            'close': 0
        })
        sample_data['high'] = sample_data['open'] + np.random.uniform(0, 2, 100)
        sample_data['low'] = sample_data['open'] - np.random.uniform(0, 2, 100)
        sample_data['close'] = sample_data['open'] + np.random.randn(100) * 0.3
    
    indicators_data = TechnicalIndicators.calculate_all_indicators(sample_data)
    indicator_summary = TechnicalIndicators.get_indicator_summary(indicators_data)
    interpretations = TechnicalIndicators.interpret_indicators(indicators_data)
    
    enhanced_results = results_df.copy()
    
    if len(indicators_data) > 0:
        latest_indicators = indicators_data.iloc[-1]
        
        enhanced_results['RSI'] = latest_indicators.get('rsi', 50)
        enhanced_results['Momentum'] = latest_indicators.get('momentum', 0)
        enhanced_results['BB_Position'] = latest_indicators.get('bb_percent_b', 50)
        enhanced_results['MACD_Signal'] = 1 if latest_indicators.get('macd_line', 0) > latest_indicators.get('macd_signal', 0) else 0
        enhanced_results['Stochastic'] = latest_indicators.get('stoch_k', 50)
        enhanced_results['Williams_R'] = latest_indicators.get('williams_r', -50)
        enhanced_results['ATR'] = latest_indicators.get('atr', 1)
        enhanced_results['CCI'] = latest_indicators.get('cci', 0)
    
    enhanced_results['RSI_Signal'] = interpretations.get('rsi', 'Neutral')
    enhanced_results['BB_Signal'] = interpretations.get('bollinger', 'Within Bands')
    enhanced_results['MACD_Trend'] = interpretations.get('macd', 'Neutral')
    enhanced_results['Stoch_Signal'] = interpretations.get('stochastic', 'Neutral')
    
    return enhanced_results, indicator_summary, interpretations

if __name__ == "__main__":
    print("🔧 Testing Technical Indicators Calculator...")
    
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 0,
        'low': 0,
        'close': 0
    })
    sample_data['high'] = sample_data['open'] + np.random.uniform(0, 2, 100)
    sample_data['low'] = sample_data['open'] - np.random.uniform(0, 2, 100)
    sample_data['close'] = sample_data['open'] + np.random.randn(100) * 0.3
    
    indicators_data = TechnicalIndicators.calculate_all_indicators(sample_data)
    summary = TechnicalIndicators.get_indicator_summary(indicators_data)
    interpretations = TechnicalIndicators.interpret_indicators(indicators_data)
    
    print("#  Technical indicators calculated successfully!")
    print(f"#  Calculated {len(indicators_data.columns) - 4} technical indicators")
    print(f"📈 Latest RSI: {indicators_data['rsi'].iloc[-1]:.2f}")
    print(f"📉 Latest MACD: {indicators_data['macd_line'].iloc[-1]:.4f}")
    print(f"🎯 Interpretations: {len(interpretations)} signals generated")

