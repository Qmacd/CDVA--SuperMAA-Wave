import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class WaveMAA_Dataset(Dataset):
    """
    Wave MAA Dataset
    Supports CD1/CD2/CD3/VA four classification tasks + fish body length prediction
    """
    
    def __init__(self, features, cd1_labels, cd2_labels, cd3_labels, va_labels,
                 cd1_lengths, cd2_lengths, cd3_lengths, va_lengths, timestamps=None, close_prices=None):
        """
        Initialize dataset
        
        Args:
            features: Feature data (N, seq_len, feature_dim)
            cd1_labels: CD1 classification labels (N,)
            cd2_labels: CD2 classification labels (N,)
            cd3_labels: CD3 classification labels (N,)
            va_labels: VA classification labels (N,)
            cd1_lengths: CD1 fish body lengths (N,)
            cd2_lengths: CD2 fish body lengths (N,)
            cd3_lengths: CD3 fish body lengths (N,)
            va_lengths: VA fish body lengths (N,)
            timestamps: Timestamp information for each sample (N,)
            close_prices: Close price for each sample (N,)
        """
        self.features = torch.FloatTensor(features)
        self.cd1_labels = torch.LongTensor(cd1_labels)
        self.cd2_labels = torch.LongTensor(cd2_labels)
        self.cd3_labels = torch.LongTensor(cd3_labels)
        self.va_labels = torch.LongTensor(va_labels)
        self.cd1_lengths = torch.FloatTensor(cd1_lengths)
        self.cd2_lengths = torch.FloatTensor(cd2_lengths)
        self.cd3_lengths = torch.FloatTensor(cd3_lengths)
        self.va_lengths = torch.FloatTensor(va_lengths)
        self.timestamps = timestamps  # 保存时间戳信息
        self.close_prices = close_prices  # 保存收盘价信息
        
        assert len(self.features) == len(self.cd1_labels) == len(self.cd2_labels) == len(self.cd3_labels) == len(self.va_labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        item = {
            'features': self.features[idx],
            'cd1_label': self.cd1_labels[idx],
            'cd2_label': self.cd2_labels[idx],
            'cd3_label': self.cd3_labels[idx],
            'va_label': self.va_labels[idx],
            'cd1_length': self.cd1_lengths[idx],
            'cd2_length': self.cd2_lengths[idx],
            'cd3_length': self.cd3_lengths[idx],
            'va_length': self.va_lengths[idx]
        }
        
        # 添加时间戳信息（转换为字符串格式）
        if self.timestamps is not None and idx < len(self.timestamps):
            timestamp = self.timestamps[idx]
            if hasattr(timestamp, 'strftime'):  # 如果是pandas Timestamp或datetime对象
                item['timestamp'] = timestamp.strftime('%Y-%m-%d')
            else:
                item['timestamp'] = str(timestamp)
        
        # 添加收盘价信息
        if self.close_prices is not None and idx < len(self.close_prices):
            item['close_price'] = self.close_prices[idx]
        
        return item

def detect_data_format(data_path):
    """
    Intelligently detect dataset type and format
    """
    print(f"Detecting data format: {data_path}")
    
    if os.path.exists(os.path.join(data_path, 'day_labeled_data')):
        dataset_type = 'day_labeled'
        data_dir = os.path.join(data_path, 'day_labeled_data', 'output_res')
        print("Detected day_labeled dataset")
    elif os.path.exists(os.path.join(data_path, 'output_res2')):
        dataset_type = '15min'
        data_dir = os.path.join(data_path, 'output_res2')
        print("Detected 15min dataset")
    elif os.path.exists(os.path.join(data_path, 'cdva_dataset')):
        dataset_type = 'cdva'
        data_dir = os.path.join(data_path, 'cdva_dataset')
        print("Detected CDVA dataset")
    else:
        csv_files = glob.glob(os.path.join(data_path, '*.csv'))
        if csv_files:
            dataset_type = 'direct_csv'
            data_dir = data_path
            print(f" Detected direct CSV files: {len(csv_files)} files")
        else:
            raise ValueError(f"No supported dataset format found in: {data_path}")
    
    return dataset_type, data_dir

def load_and_process_csv(file_path, dataset_type):
    """
    Load and process single CSV file
    """
    try:
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'utf-8-sig']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        
        if df is None:
            print(f"  Could not read {os.path.basename(file_path)} with any encoding")
            return None
        
        if len(df) == 0:
            print(f"  File {os.path.basename(file_path)} is empty")
            return None
        
        df.columns = df.columns.str.lower().str.strip()
        
        column_mapping = {
            'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close',
            'openinterest': 'open_interest', 'volume': 'volume',
            'open_price': 'open', 'high_price': 'high', 
            'low_price': 'low', 'close_price': 'close'
        }
        
        df = df.rename(columns=column_mapping)
        
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"  File {os.path.basename(file_path)} missing required columns: {missing_cols}")
            print(f"    Available columns: {list(df.columns)}")
            return None
        
        if dataset_type == 'cdva':
            label_cols = ['cdva_label_cd_1', 'cdva_label_cd_2', 'cdva_label_cd_3', 'cdva_label_va']
            missing_labels = [col for col in label_cols if col not in df.columns]
            
            if missing_labels:
                print(f"  CDVA file {os.path.basename(file_path)} missing label columns: {missing_labels}")
                for col in missing_labels:
                    df[col] = 'S'  # Default to short fish
        else:
            print(f"  Generating labels for {os.path.basename(file_path)}...")
            df = generate_labels_for_new_data(df)
        
        ema_cols = ['ema5', 'ema10', 'ema20', 'ema30', 'ema60', 'ema120']
        for ema_col in ema_cols:
            if ema_col not in df.columns:
                period = int(ema_col[3:])  # Extract period number
                df[ema_col] = df['close'].ewm(span=period).mean()
        
        required_label_cols = ['cdva_label_cd_1', 'cdva_label_cd_2', 'cdva_label_cd_3', 'cdva_label_va']
        for col in required_label_cols:
            if col not in df.columns:
                df[col] = 'S'  # Default value
        
        initial_len = len(df)
        
        critical_cols = ['open', 'high', 'low', 'close']
        df = df.dropna(subset=critical_cols)
        
        for col in required_label_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna('S')  # Forward fill, finally fill with 'S'
        
        for ema_col in ema_cols:
            if ema_col in df.columns:
                df[ema_col] = df[ema_col].fillna(method='ffill')
        
        required_columns = ['open', 'high', 'low', 'close'] + ema_cols + required_label_cols
        
        for col in required_columns:
            if col not in df.columns:
                if col in ema_cols:
                    df[col] = df['close']  # Use close price as fallback for EMA
                elif col in required_label_cols:
                    df[col] = 'S'
                else:
                    df[col] = 0.0
        
        available_required_cols = [col for col in required_columns if col in df.columns]
        other_cols = [col for col in df.columns if col not in required_columns]
        df = df[available_required_cols + other_cols]
        
        final_len = len(df)
        
        if final_len < initial_len:
            print(f" {os.path.basename(file_path)}: Processed {initial_len - final_len} rows, remaining {final_len} rows")
        
        if len(df) < 50:  # Skip only if data is too little
            print(f" {os.path.basename(file_path)} too little data ({len(df)} rows), skipping")
            return None
        
        print(f" Successfully processed {os.path.basename(file_path)}, data length: {len(df)}")
        
        return df
        
    except Exception as e:
        print(f" Error processing file {file_path}: {type(e).__name__}: {e}")
        return None

def generate_labels_for_new_data(df):
    """
    Generate CDVA labels for new dataset
    Generate reasonable labels based on price changes and technical indicators
    """
    df['price_change'] = df['close'].pct_change()
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_change_10'] = df['close'].pct_change(10)
    df['price_change_20'] = df['close'].pct_change(20)
    
    df['volatility_5'] = df['price_change'].rolling(5).std()
    df['volatility_10'] = df['price_change'].rolling(10).std()
    df['volatility_20'] = df['price_change'].rolling(20).std()
    
    cd1_threshold = df['price_change_5'].quantile(0.6)  # 60th percentile
    df['cdva_label_cd_1'] = np.where(df['price_change_5'] > cd1_threshold, 'L', 'S')
    
    cd2_threshold = df['price_change_10'].quantile(0.6)
    df['cdva_label_cd_2'] = np.where(df['price_change_10'] > cd2_threshold, 'L', 'S')
    
    cd3_threshold = df['price_change_20'].quantile(0.6)
    df['cdva_label_cd_3'] = np.where(df['price_change_20'] > cd3_threshold, 'L', 'S')
    
    va_score = (
        (df['price_change_5'] > df['price_change_5'].quantile(0.5)).astype(int) +
        (df['price_change_10'] > df['price_change_10'].quantile(0.5)).astype(int) +
        (df['price_change_20'] > df['price_change_20'].quantile(0.5)).astype(int) +
        (df['volatility_10'] < df['volatility_10'].quantile(0.5)).astype(int)  # Low volatility is good
    )
    df['cdva_label_va'] = np.where(va_score >= 2, 'L', 'S')
    
    return df

def create_sliding_windows(data, window_size=20):
    """
    Create sliding window data
    """
    features = []
    labels = {
        'cd1': [], 'cd2': [], 'cd3': [], 'va': []
    }
    timestamps = []  # 收集时间戳信息
    close_prices = []  # 收集收盘价信息
    
    feature_cols = ['open', 'high', 'low', 'close', 'ema5', 'ema10', 'ema20', 'ema30', 'ema60', 'ema120']
    
    available_cols = [col for col in feature_cols if col in data.columns]
    if len(available_cols) < 4:  # At least need OHLC
        raise ValueError(f"Data missing required feature columns, available columns: {available_cols}")
    
    feature_data = data[available_cols].values
    
    label_mapping = {'S': 0, 'L': 1}
    cd1_labels = [label_mapping.get(x, 0) for x in data['cdva_label_cd_1'].values]
    cd2_labels = [label_mapping.get(x, 0) for x in data['cdva_label_cd_2'].values]
    cd3_labels = [label_mapping.get(x, 0) for x in data['cdva_label_cd_3'].values]
    va_labels = [label_mapping.get(x, 0) for x in data['cdva_label_va'].values]
    
    for i in range(window_size, len(feature_data)):
        window_features = feature_data[i-window_size:i]
        features.append(window_features)
        
        labels['cd1'].append(cd1_labels[i])
        labels['cd2'].append(cd2_labels[i])
        labels['cd3'].append(cd3_labels[i])
        labels['va'].append(va_labels[i])
        
        # 收集对应的时间戳（窗口结束时间）
        if 'timestamp' in data.columns:
            timestamps.append(data['timestamp'].iloc[i])
        else:
            # 如果没有时间戳，使用索引作为时间标识
            timestamps.append(f"sample_{i}")
        
        # 收集对应的收盘价（窗口结束时间）
        if 'close' in data.columns:
            close_prices.append(data['close'].iloc[i])
        else:
            # 如果没有收盘价，使用0作为默认值
            close_prices.append(0.0)
    
    print(f" Created sliding windows with unified timestamps")
    print(f"   Features used: {available_cols}")
    if 'timestamp' in data.columns:
        window_start_date = data['timestamp'].iloc[window_size-1] if len(data) > window_size else data['timestamp'].iloc[0]
        window_end_date = data['timestamp'].iloc[-1] if len(data) > 0 else "N/A"
        print(f"   Window date range: {window_start_date.strftime('%Y-%m-%d')} to {window_end_date.strftime('%Y-%m-%d')}")
    
    return np.array(features), labels, available_cols, timestamps, close_prices

def generate_fish_lengths(labels, task_type):
    """
    Generate fish body length data
    Generate reasonable length values based on labels and task type
    """
    lengths = []
    
    length_ranges = {
        'cd1': (2, 8),    # Short-term: 2-8 days
        'cd2': (5, 15),   # Medium-term: 5-15 days
        'cd3': (10, 25),  # Long-term: 10-25 days
        'va': (3, 20)     # Comprehensive: 3-20 days
    }
    
    min_len, max_len = length_ranges.get(task_type, (2, 10))
    
    for label in labels:
        if label == 1:  # L class (long fish)
            length = np.random.randint(min_len + (max_len - min_len) // 2, max_len + 1)
        else:  # S class (short fish)
            length = np.random.randint(min_len, min_len + (max_len - min_len) // 2 + 1)
        
        lengths.append(float(length))
    
    return lengths

def load_wave_maa_data(data_path='.', window_size=20, test_split=0.15, val_split=0.15, max_files=None):
    """
    Load Wave MAA data with length standardization
    
    Args:
        data_path: Data path
        window_size: Sliding window size
        test_split: Test set ratio
        val_split: Validation set ratio
        max_files: Maximum number of files (None means load all files)
    
    Returns:
        train_dataset, val_dataset, test_dataset, scaler, feature_columns
    """
    print(" Starting to load Wave MAA data...")
    
    dataset_type, data_dir = detect_data_format(data_path)
    
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    new_data_dir = os.path.join(data_dir, 'new_data')
    if os.path.exists(new_data_dir):
        new_data_files = glob.glob(os.path.join(new_data_dir, '*.csv'))
        print(f" Found {len(new_data_files)} files in new_data directory")
        csv_files.extend(new_data_files)
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    print(f" Total found {len(csv_files)} data files")
    
    if max_files is not None and len(csv_files) > max_files:
        main_files = glob.glob(os.path.join(data_dir, '*.csv'))
        new_files = [f for f in csv_files if f not in main_files]
        
        if len(main_files) >= max_files:
            csv_files = main_files[:max_files]
        else:
            remaining = max_files - len(main_files)
            csv_files = main_files + new_files[:remaining]
        
        print(f" Limited to {max_files} files: {len(main_files)} main + {len(csv_files)-len(main_files)} new_data")
    
    print("\n Performing comprehensive data length analysis...")
    standard_length, length_stats, coverage_stats = analyze_data_lengths(csv_files, sample_size=min(100, len(csv_files)))
    
    min_required_length = window_size + 50
    if standard_length < min_required_length:
        print(f"  Standard length {standard_length} too small, using minimum {min_required_length}")
        standard_length = min_required_length
    
    if max_files is not None:
        usable_count = 0
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                if len(df) >= standard_length:
                    usable_count += 1
            except:
                continue
        
        usable_ratio = usable_count / len(csv_files) if csv_files else 0
        print(f" Current standard {standard_length} would use {usable_count}/{len(csv_files)} files ({usable_ratio*100:.1f}%)")
        
        if usable_ratio < 0.5:  # Less than 50% usable
            print(f"  Too few files usable with standard {standard_length}, adjusting...")
            
            actual_lengths = []
            for file_path in csv_files:
                try:
                    df = pd.read_csv(file_path)
                    actual_lengths.append(len(df))
                except:
                    continue
            
            if actual_lengths:
                adjusted_standard = int(np.percentile(actual_lengths, 50))
                adjusted_standard = max(adjusted_standard, min_required_length)
                
                print(f" Adjusted standard length: {standard_length} → {adjusted_standard}")
                standard_length = adjusted_standard
                
                final_usable = sum(1 for length in actual_lengths if length >= adjusted_standard)
                final_ratio = final_usable / len(actual_lengths)
                print(f" After adjustment: {final_usable}/{len(actual_lengths)} files usable ({final_ratio*100:.1f}%)")
    
    all_data = []
    processed_files = 0
    skipped_files = 0
    operation_counts = {
        'no_change': 0,
        'truncated': 0,
        'too_short_skipped': 0,
        'load_failed': 0,  # Track files that failed to load
        'processing_error': 0  # Track processing errors
    }
    
    error_details = {
        'file_load_errors': [],
        'too_short_files': [],
        'processing_errors': []
    }
    
    for i, file_path in enumerate(csv_files):
        print(f" Processing file {i+1}/{len(csv_files)}: {os.path.basename(file_path)}")
        
        df = load_and_process_csv(file_path, dataset_type)
        if df is not None:
            try:
                standardized_df, operation = standardize_dataframe_length(
                    df, standard_length, min_length=standard_length  # Set min_length = standard_length to skip shorter files
                )
                
                standardized_df = add_unified_timestamp(standardized_df, standard_length)
                
                if 'truncated' in operation:
                    operation_counts['truncated'] += 1
                else:
                    operation_counts['no_change'] += 1
                
                if len(standardized_df) == standard_length:
                    all_data.append(standardized_df)
                    processed_files += 1
                    print(f"    {operation}, added timestamp, final length: {len(standardized_df)}")
                else:
                    print(f"    Length error: expected {standard_length}, got {len(standardized_df)}")
                    skipped_files += 1
                    operation_counts['processing_error'] += 1
                    error_details['processing_errors'].append({
                        'file': os.path.basename(file_path),
                        'expected': standard_length,
                        'actual': len(standardized_df),
                        'reason': 'Length mismatch after processing'
                    })
                    
            except ValueError as e:
                if "too short for standardization" in str(e):
                    print(f"   Skipped: file too short ({len(df)} < {standard_length})")
                    operation_counts['too_short_skipped'] += 1
                    error_details['too_short_files'].append({
                        'file': os.path.basename(file_path),
                        'length': len(df),
                        'required': standard_length
                    })
                else:
                    print(f"    Skipped: {e}")
                    operation_counts['processing_error'] += 1
                    error_details['processing_errors'].append({
                        'file': os.path.basename(file_path),
                        'error': str(e),
                        'reason': 'Processing ValueError'
                    })
                skipped_files += 1
        else:
            skipped_files += 1
            operation_counts['load_failed'] += 1
            error_details['file_load_errors'].append({
                'file': os.path.basename(file_path),
                'reason': 'load_and_process_csv returned None'
            })
        
        if max_files is not None and processed_files >= max_files:
            print(f"  Reached max_files limit ({max_files}), stopping")
            break
    
    if not all_data:
        raise ValueError("No data files processed successfully")
    
    print(f"\n Data processing summary:")
    print(f"   Successfully processed: {processed_files} files")
    print(f"   Skipped files: {skipped_files} files")
    print(f"   Operations performed:")
    print(f"     - No change needed: {operation_counts['no_change']} files")
    print(f"     - Truncated (too long): {operation_counts['truncated']} files")
    print(f"     - Skipped (too short): {operation_counts['too_short_skipped']} files")
    print(f"     - Load failed: {operation_counts['load_failed']} files")
    print(f"     - Processing errors: {operation_counts['processing_error']} files")
    print(f"   Final standard length: {standard_length} rows per file")
    print(f"   Timestamp range: 2000-01-01 to {standard_length} days")
    
    if error_details['file_load_errors']:
        print(f"\n File Load Errors ({len(error_details['file_load_errors'])} files):")
        for i, error in enumerate(error_details['file_load_errors'][:5]):  # Show first 5
            print(f"   {i+1}. {error['file']}: {error['reason']}")
        if len(error_details['file_load_errors']) > 5:
            print(f"   ... and {len(error_details['file_load_errors'])-5} more files")
    
    if error_details['too_short_files']:
        print(f"\n Too Short Files ({len(error_details['too_short_files'])} files):")
        for i, error in enumerate(error_details['too_short_files'][:5]):  # Show first 5
            print(f"   {i+1}. {error['file']}: {error['length']} rows < {error['required']} required")
        if len(error_details['too_short_files']) > 5:
            print(f"   ... and {len(error_details['too_short_files'])-5} more files")
    
    if error_details['processing_errors']:
        print(f"\n Processing Errors ({len(error_details['processing_errors'])} files):")
        for i, error in enumerate(error_details['processing_errors'][:5]):  # Show first 5
            print(f"   {i+1}. {error['file']}: {error.get('error', error.get('reason', 'Unknown error'))}")
        if len(error_details['processing_errors']) > 5:
            print(f"   ... and {len(error_details['processing_errors'])-5} more files")
    
    print(f"\n Verifying data consistency...")
    is_consistent, consistency_report = verify_data_consistency(all_data, standard_length)
    
    if not is_consistent:
        error_msg = f"Data consistency verification failed. Issues found in {consistency_report['total_files']} files."
        if consistency_report['length_issues']:
            error_msg += f" Length issues: {len(consistency_report['length_issues'])} files."
        print(f" {error_msg}")
        raise ValueError(error_msg)
    
    print(f" Data consistency verification passed!")
    
    print(" Merging standardized data with unified timestamps...")
    combined_data = pd.concat(all_data, ignore_index=True)
    expected_total_length = len(all_data) * standard_length
    actual_total_length = len(combined_data)
    
    if 'timestamp' in combined_data.columns:
        min_date = combined_data['timestamp'].min()
        max_date = combined_data['timestamp'].max()
        print(f" Timestamp range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    print(f" Data merging results:")
    print(f"   Expected total length: {expected_total_length:,} rows")
    print(f"   Actual total length: {actual_total_length:,} rows")
    print(f"   Files merged: {len(all_data)}")
    print(f"   Rows per file: {standard_length}")
    print(f"   Unified timestamp: 2000-01-01 start date")
    
    if actual_total_length != expected_total_length:
        raise ValueError(f"Data merging error: expected {expected_total_length} rows, got {actual_total_length}")
    
    print(f" Data merging successful with unified timestamps!")
    
    print(f" Creating sliding windows, window size: {window_size}")
    features, labels, feature_columns, timestamps, close_prices = create_sliding_windows(combined_data, window_size)
    
    print(f" Created {len(features)} time windows")
    print(f"   Feature shape: {features.shape}")
    print(f"   Feature columns: {feature_columns}")
    
    print(" Standardizing feature data...")
    scaler = StandardScaler()
    
    original_shape = features.shape
    features_reshaped = features.reshape(-1, features.shape[-1])
    features_scaled = scaler.fit_transform(features_reshaped)
    features = features_scaled.reshape(original_shape)
    
    print(" Generating fish body length data...")
    cd1_lengths = generate_fish_lengths(labels['cd1'], 'cd1')
    cd2_lengths = generate_fish_lengths(labels['cd2'], 'cd2')
    cd3_lengths = generate_fish_lengths(labels['cd3'], 'cd3')
    va_lengths = generate_fish_lengths(labels['va'], 'va')
    
    print(" Label distribution:")
    for task in ['cd1', 'cd2', 'cd3', 'va']:
        task_labels = labels[task]
        l_count = sum(1 for x in task_labels if x == 1)
        s_count = sum(1 for x in task_labels if x == 0)
        print(f"   {task.upper()}: L={l_count}, S={s_count}")
    
    print(" Fish body length statistics:")
    for task, lengths in [('cd1', cd1_lengths), ('cd2', cd2_lengths), ('cd3', cd3_lengths), ('va', va_lengths)]:
        avg_len = np.mean(lengths)
        min_len = np.min(lengths)
        max_len = np.max(lengths)
        print(f"   {task.upper()}: Average={avg_len:.1f}days, Range={min_len:.0f}-{max_len:.0f}days")
    
    n_samples = len(features)
    test_size = int(n_samples * test_split)
    val_size = int(n_samples * val_split)
    train_size = n_samples - test_size - val_size
    
    train_features = features[:train_size]
    val_features = features[train_size:train_size + val_size]
    test_features = features[train_size + val_size:]
    
    train_labels = {k: v[:train_size] for k, v in labels.items()}
    val_labels = {k: v[train_size:train_size + val_size] for k, v in labels.items()}
    test_labels = {k: v[train_size + val_size:] for k, v in labels.items()}
    
    train_cd1_lengths = cd1_lengths[:train_size]
    train_cd2_lengths = cd2_lengths[:train_size]
    train_cd3_lengths = cd3_lengths[:train_size]
    train_va_lengths = va_lengths[:train_size]
    
    val_cd1_lengths = cd1_lengths[train_size:train_size + val_size]
    val_cd2_lengths = cd2_lengths[train_size:train_size + val_size]
    val_cd3_lengths = cd3_lengths[train_size:train_size + val_size]
    val_va_lengths = va_lengths[train_size:train_size + val_size]
    
    test_cd1_lengths = cd1_lengths[train_size + val_size:]
    test_cd2_lengths = cd2_lengths[train_size + val_size:]
    test_cd3_lengths = cd3_lengths[train_size + val_size:]
    test_va_lengths = va_lengths[train_size + val_size:]
    
    # 分割时间戳
    train_timestamps = timestamps[:train_size]
    val_timestamps = timestamps[train_size:train_size + val_size]
    test_timestamps = timestamps[train_size + val_size:]
    
    # 分割收盘价
    train_close_prices = close_prices[:train_size]
    val_close_prices = close_prices[train_size:train_size + val_size]
    test_close_prices = close_prices[train_size + val_size:]
    
    print(" Data splitting:")
    print(f"   Training set: {len(train_features)} samples")
    print(f"   Validation set: {len(val_features)} samples")
    print(f"   Test set: {len(test_features)} samples")
    
    print(f"\n Final data consistency check:")
    print(f"   All training features shape: {train_features.shape}")
    print(f"   All validation features shape: {val_features.shape}")
    print(f"   All test features shape: {test_features.shape}")
    print(f"   Total samples: {len(train_features) + len(val_features) + len(test_features)}")
    
    train_dataset = WaveMAA_Dataset(
        train_features, train_labels['cd1'], train_labels['cd2'], train_labels['cd3'], train_labels['va'],
        train_cd1_lengths, train_cd2_lengths, train_cd3_lengths, train_va_lengths, train_timestamps, train_close_prices
    )
    
    val_dataset = WaveMAA_Dataset(
        val_features, val_labels['cd1'], val_labels['cd2'], val_labels['cd3'], val_labels['va'],
        val_cd1_lengths, val_cd2_lengths, val_cd3_lengths, val_va_lengths, val_timestamps, val_close_prices
    )
    
    test_dataset = WaveMAA_Dataset(
        test_features, test_labels['cd1'], test_labels['cd2'], test_labels['cd3'], test_labels['va'],
        test_cd1_lengths, test_cd2_lengths, test_cd3_lengths, test_va_lengths, test_timestamps, test_close_prices
    )
    
    print(f" All datasets created successfully with consistent dimensions")
    
    return train_dataset, val_dataset, test_dataset, scaler, feature_columns

def create_wave_maa_data_loaders(train_dataset, val_dataset, test_dataset, 
                                batch_size=32, num_workers=0):
    """
    Create data loaders
    """
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def analyze_data_lengths(csv_files, sample_size=100):
    """
    Analyze data lengths from CSV files to determine optimal standard length
    
    Args:
        csv_files: List of CSV file paths
        sample_size: Number of files to analyze for length distribution
    
    Returns:
        standard_length: Recommended standard length for all files
        length_stats: Dictionary with length statistics
    """
    print(f" Analyzing data lengths from {min(sample_size, len(csv_files))} files...")
    
    data_lengths = []
    valid_files = 0
    
    sample_files = csv_files[:min(sample_size, len(csv_files))]
    
    for i, file_path in enumerate(sample_files):
        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                data_lengths.append(len(df))
                valid_files += 1
            
            if (i + 1) % 20 == 0:
                print(f"   Analyzed {i+1}/{len(sample_files)} files...")
                
        except Exception as e:
            print(f"  Could not read {os.path.basename(file_path)}: {e}")
            continue
    
    if not data_lengths:
        raise ValueError("Could not analyze any data files")
    
    data_lengths = np.array(data_lengths)
    length_stats = {
        'count': len(data_lengths),
        'min': np.min(data_lengths),
        'max': np.max(data_lengths),
        'mean': np.mean(data_lengths),
        'median': np.median(data_lengths),
        'std': np.std(data_lengths),
        'q25': np.percentile(data_lengths, 25),
        'q75': np.percentile(data_lengths, 75),
        'q90': np.percentile(data_lengths, 90)
    }
    
    print(f"Finding the optimal length using 'most common AND longest possible' strategy with 70% coverage guarantee...")
    
    from collections import Counter
    
    range_size = 50  # 每个区间的大小
    length_ranges = {}
    
    for length in data_lengths:
        range_key = (length // range_size) * range_size
        if range_key not in length_ranges:
            length_ranges[range_key] = []
        length_ranges[range_key].append(length)
    
    range_stats = {}
    for range_key, lengths in length_ranges.items():
        range_stats[range_key] = {
            'count': len(lengths),
            'percentage': len(lengths) / len(data_lengths) * 100,
            'min_length': min(lengths),
            'max_length': max(lengths),
            'mean_length': np.mean(lengths),
            'median_length': np.median(lengths),
            'lengths': sorted(lengths)
        }
    
    sorted_ranges = sorted(range_stats.items(), 
                          key=lambda x: (-x[1]['count'], -x[1]['max_length']))
    
    print(f"\n Length range distribution analysis (top 10):")
    for i, (range_key, stats) in enumerate(sorted_ranges[:10]):
        print(f"   #{i+1}: Range {range_key}-{range_key+range_size-1}")
        print(f"        Files: {stats['count']} ({stats['percentage']:.1f}%)")
        print(f"        Length span: {stats['min_length']}-{stats['max_length']}")
        if i == 0:
            print(f"        ← SELECTED: Most files + good length range")
    
    best_range_key, best_range_stats = sorted_ranges[0]
    print(f"\n Selected range {best_range_key}-{best_range_key+range_size-1}:")
    print(f"   Files in range: {best_range_stats['count']} ({best_range_stats['percentage']:.1f}%)")
    print(f"   Length span: {best_range_stats['min_length']} - {best_range_stats['max_length']}")
    
    range_lengths = best_range_stats['lengths']
    target_length_75th = int(np.percentile(range_lengths, 75))
    target_length_90th = int(np.percentile(range_lengths, 90))
    target_length_max = max(range_lengths)
    
    print(f"   Length percentiles in range:")
    print(f"     50th (median): {int(best_range_stats['median_length'])}")
    print(f"     75th: {target_length_75th}")
    print(f"     90th: {target_length_90th}")
    print(f"     Maximum: {target_length_max}")
    
    target_length = target_length_75th
    
    usable_files = np.sum(data_lengths >= target_length)
    usable_percentage = usable_files / len(data_lengths) * 100
    
    print(f"\n Coverage analysis for length {target_length}:")
    print(f"   Files usable (>= {target_length}): {usable_files} ({usable_percentage:.1f}%)")
    
    if usable_percentage < 70.0:
        print(f"  Coverage {usable_percentage:.1f}% is too low (target: >=70%), adjusting...")
        
        alternative_length = int(best_range_stats['median_length'])
        alternative_coverage = np.sum(data_lengths >= alternative_length) / len(data_lengths) * 100
        print(f"   Alternative A (range median {alternative_length}): {alternative_coverage:.1f}% coverage")
        
        if alternative_coverage >= 70.0:
            target_length = alternative_length
            usable_percentage = alternative_coverage
            print(f"   → Adopted Alternative A (meets 70% threshold)")
        else:
            sorted_lengths = np.sort(data_lengths)[::-1]  # 降序
            target_index = int(len(sorted_lengths) * 0.3)  # 30%位置意味着70%覆盖
            fallback_length = int(sorted_lengths[target_index])
            fallback_coverage = 70.0  # 按定义
            print(f"   Alternative B (70% coverage threshold): length {fallback_length}")
            
            target_length = fallback_length
            usable_percentage = fallback_coverage
            print(f"   → Adopted Alternative B (guaranteed 70% coverage)")
            
        if usable_percentage < 70.0:
            global_70_coverage_length = int(np.percentile(data_lengths, 30))  # 30%分位数意味着70%文件>=此长度
            global_coverage = np.sum(data_lengths >= global_70_coverage_length) / len(data_lengths) * 100
            print(f"   Alternative C (global 30th percentile): length {global_70_coverage_length}, coverage {global_coverage:.1f}%")
            
            target_length = global_70_coverage_length
            usable_percentage = global_coverage
            print(f"   → Adopted Alternative C (global strategy)")
    
    print(f"\n Final target length: {target_length}")
    print(f"   Expected usable files: {np.sum(data_lengths >= target_length)} ({usable_percentage:.1f}%)")
    
    standard_length = target_length
    
    coverage_stats = {
        'exact_match': np.sum(data_lengths == standard_length) / len(data_lengths) * 100,
        'can_use_full': np.sum(data_lengths >= standard_length) / len(data_lengths) * 100,
        'will_be_skipped': np.sum(data_lengths < standard_length) / len(data_lengths) * 100,
        'most_common_range': f"{best_range_key}-{best_range_key+range_size-1}",
        'files_in_common_range': best_range_stats['count'],
        'range_percentage': best_range_stats['percentage']
    }
    
    print(f"\n Data length analysis results:")
    print(f"   Files analyzed: {length_stats['count']}")
    print(f"   Length range: {length_stats['min']:.0f} - {length_stats['max']:.0f}")
    print(f"   Mean: {length_stats['mean']:.1f}, Median: {length_stats['median']:.1f}")
    print(f"   25th-75th percentile: {length_stats['q25']:.0f} - {length_stats['q75']:.0f}")
    print(f"   Most common range: {coverage_stats['most_common_range']} ({coverage_stats['files_in_common_range']} files)")
    
    print(f"\n Selected standard length: {standard_length}")
    print(f"   Files with exact length: {coverage_stats['exact_match']:.1f}%")
    print(f"   Files usable (>= standard): {coverage_stats['can_use_full']:.1f}%")
    print(f"   Files will be skipped: {coverage_stats['will_be_skipped']:.1f}%")
    print(f"   Strategy: Most common range + 70% coverage guarantee")
    return standard_length, length_stats, coverage_stats

def standardize_dataframe_length(df, target_length, min_length=100):
    """
    Standardize a dataframe to target length
    
    Args:
        df: Input dataframe
        target_length: Target number of rows
        min_length: Minimum acceptable length (ignored - all files shorter than target are skipped)
    
    Returns:
        standardized_df: Dataframe with exactly target_length rows
        operation: String describing what operation was performed
    """
    current_length = len(df)
    
    if current_length == target_length:
        return df.copy(), "no_change"
    
    elif current_length > target_length:
        standardized_df = df.tail(target_length).reset_index(drop=True)
        return standardized_df, f"truncated_from_{current_length}"
    
    else:
        raise ValueError(f"too short for standardization: {current_length} < {target_length}")

def verify_data_consistency(all_data, expected_length):
    """
    Verify that all dataframes have consistent length and structure
    
    Args:
        all_data: List of pandas dataframes
        expected_length: Expected number of rows for each dataframe
    
    Returns:
        is_consistent: Boolean indicating if all data is consistent
        consistency_report: Dictionary with detailed consistency information
    """
    print(f"\n Performing comprehensive data consistency verification...")
    
    consistency_report = {
        'total_files': len(all_data),
        'expected_length': expected_length,
        'length_issues': [],
        'column_issues': [],
        'data_type_issues': [],
        'is_consistent': True
    }
    
    if not all_data:
        consistency_report['is_consistent'] = False
        return False, consistency_report
    
    core_required_columns = ['open', 'high', 'low', 'close', 'ema5', 'ema10', 'ema20', 'ema30', 'ema60', 'ema120',
                           'cdva_label_cd_1', 'cdva_label_cd_2', 'cdva_label_cd_3', 'cdva_label_va', 'timestamp', 'date']
    
    for i, df in enumerate(all_data):
        if len(df) != expected_length:
            consistency_report['length_issues'].append({
                'file_index': i,
                'actual_length': len(df),
                'expected_length': expected_length
            })
            consistency_report['is_consistent'] = False
        
        missing_core_cols = [col for col in core_required_columns if col not in df.columns]
        if missing_core_cols:
            consistency_report['column_issues'].append({
                'file_index': i,
                'missing_columns': missing_core_cols,
                'extra_columns': []
            })
            print(f"  File {i} missing core columns: {missing_core_cols}")
    
    print(f" Standardizing column structure across all files...")
    
    all_columns = set()
    for df in all_data:
        all_columns.update(df.columns)
    
    priority_columns = ['timestamp', 'date', 'open', 'high', 'low', 'close', 
                       'ema5', 'ema10', 'ema20', 'ema30', 'ema60', 'ema120',
                       'cdva_label_cd_1', 'cdva_label_cd_2', 'cdva_label_cd_3', 'cdva_label_va']
    
    other_columns = sorted([col for col in all_columns if col not in priority_columns])
    final_column_order = priority_columns + other_columns
    
    standardized_data = []
    for i, df in enumerate(all_data):
        df_copy = df.copy()
        
        for col in final_column_order:
            if col not in df_copy.columns:
                if col in ['ema5', 'ema10', 'ema20', 'ema30', 'ema60', 'ema120']:
                    df_copy[col] = df_copy['close'] if 'close' in df_copy.columns else 0.0
                elif col in ['cdva_label_cd_1', 'cdva_label_cd_2', 'cdva_label_cd_3', 'cdva_label_va']:
                    df_copy[col] = 'S'  # Default label
                elif col in ['timestamp', 'date']:
                    from datetime import datetime, timedelta
                    start_date = datetime.strptime('2000-01-01', '%Y-%m-%d')
                    df_copy['timestamp'] = [start_date + timedelta(days=j) for j in range(len(df_copy))]
                    df_copy['date'] = [ts.strftime('%Y-%m-%d') for ts in df_copy['timestamp']]
                else:
                    df_copy[col] = 0.0
        
        available_columns = [col for col in final_column_order if col in df_copy.columns]
        df_copy = df_copy[available_columns]
        
        standardized_data.append(df_copy)
    
    all_data.clear()
    all_data.extend(standardized_data)
    
    if len(all_data) > 0:
        reference_columns = list(all_data[0].columns)
        all_consistent = True
        
        for i, df in enumerate(all_data[1:], 1):
            if list(df.columns) != reference_columns:
                print(f"  Warning: File {i} has different column order, reordering...")
                all_data[i] = df[reference_columns]
        
        print(f" All {len(all_data)} files now have consistent structure:")
        print(f"   - Length: {expected_length} rows each")
        print(f"   - Columns: {len(reference_columns)} columns")
        print(f"   - Column order: {reference_columns[:5]}... (+{len(reference_columns)-5} more)")
        print(f"   - Total data points: {len(all_data) * expected_length:,}")
        
        consistency_report['is_consistent'] = True
    
    return consistency_report['is_consistent'], consistency_report

def add_unified_timestamp(df, standard_length, start_date='2000-01-01'):
    """
    Add unified timestamp column starting from specified date
    
    Args:
        df: Input dataframe
        standard_length: Expected length of the dataframe
        start_date: Start date for timestamp (default: '2000-01-01')
    
    Returns:
        df_with_timestamp: Dataframe with added timestamp column
    """
    from datetime import datetime, timedelta
    
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    
    timestamps = []
    for i in range(len(df)):
        timestamp = start_datetime + timedelta(days=i)
        timestamps.append(timestamp)
    
    df_copy = df.copy()
    df_copy['timestamp'] = timestamps
    df_copy['date'] = [ts.strftime('%Y-%m-%d') for ts in timestamps]
    
    return df_copy

if __name__ == '__main__':
    print("Testing data loader...")
    
    try:
        train_dataset, val_dataset, test_dataset, scaler, feature_columns = load_wave_maa_data(
            data_path='.', max_files=5
        )
        
        train_loader, val_loader, test_loader = create_wave_maa_data_loaders(
            train_dataset, val_dataset, test_dataset, batch_size=16
        )
        
        sample_batch = next(iter(train_loader))
        print(f" Data loading test successful")
        print(f"   Batch size: {sample_batch['features'].shape[0]}")
        print(f"   Feature shape: {sample_batch['features'].shape}")
        print(f"   Label shape: {sample_batch['cd1_label'].shape}")
        
    except Exception as e:
        print(f" Data loading test failed: {e}")
        import traceback
        traceback.print_exc()

