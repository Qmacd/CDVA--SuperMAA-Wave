import torch
import numpy as np
import pandas as pd
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class WaveDirectionAnalyzer:
    """Analyze wave direction and prediction confidence"""
    
    def __init__(self):
        self.wave_types = {
            0: "Downward Wave",
            1: "Upward Wave"
        }
        
    def analyze_wave_direction(self, price_sequence):
        """
        Analyze wave direction based on price sequence
        
        Args:
            price_sequence: Sequence of prices
            
        Returns:
            Dictionary with wave analysis
        """
        if len(price_sequence) < 2:
            return {
                'direction': 0,
                'direction_name': "Downward Wave",
                'price_change': 0.0,
                'price_change_pct': 0.0,
                'confidence_score': 0.5
            }
        
        start_price = price_sequence[0]
        end_price = price_sequence[-1]
        price_change = end_price - start_price
        price_change_pct = (price_change / start_price) * 100 if start_price != 0 else 0
        
        direction = 1 if price_change > 0 else 0
        
        confidence_score = min(abs(price_change_pct) / 5.0, 1.0)  # Normalize to 0-1
        confidence_score = max(confidence_score, 0.1)  # Minimum confidence
        
        return {
            'direction': direction,
            'direction_name': self.wave_types[direction],
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'confidence_score': confidence_score
        }
    
    def generate_wave_predictions(self, model_outputs, price_data=None):
        """
        Generate wave direction predictions with confidence
        
        Args:
            model_outputs: Model prediction outputs
            price_data: Optional price data for wave analysis
            
        Returns:
            Dictionary with wave predictions
        """
        predictions = {}
        
        for task in ['cd1', 'cd2', 'cd3', 'va']:
            if task in model_outputs:
                logits = model_outputs[task]
                
                probabilities = torch.softmax(logits, dim=-1)
                
                predicted_classes = torch.argmax(probabilities, dim=-1)
                
                confidence_scores = torch.max(probabilities, dim=-1)[0]
                
                predicted_classes = predicted_classes.cpu().numpy()
                confidence_scores = confidence_scores.cpu().numpy()
                
                wave_directions = []
                wave_confidences = []
                
                for i, (pred_class, confidence) in enumerate(zip(predicted_classes, confidence_scores)):
                    if price_data is not None and i < len(price_data):
                        wave_analysis = self.analyze_wave_direction(price_data[i])
                        wave_direction = wave_analysis['direction']
                        wave_confidence = wave_analysis['confidence_score']
                    else:
                        wave_direction = 1 if pred_class == 1 else 0  # Long fish -> Upward wave
                        wave_confidence = float(confidence)
                    
                    wave_directions.append(wave_direction)
                    wave_confidences.append(wave_confidence)
                
                predictions[task] = {
                    'predicted_classes': predicted_classes,
                    'class_confidences': confidence_scores,
                    'wave_directions': np.array(wave_directions),
                    'wave_confidences': np.array(wave_confidences)
                }
        
        return predictions

class PredictionSequenceGenerator:
    """Generate detailed prediction sequences for test set"""
    
    def __init__(self, model, test_loader, device='cpu'):
        """
        Initialize prediction generator
        
        Args:
            model: Trained model
            test_loader: Test data loader
            device: Computing device
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.wave_analyzer = WaveDirectionAnalyzer()
        
    def generate_predictions(self):
        """
        Generate comprehensive predictions for test set
        
        Returns:
            DataFrame with detailed predictions
        """
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(self.test_loader):
                features = features.to(self.device)
                
                outputs = self.model(features)
                
                wave_predictions = self.wave_analyzer.generate_wave_predictions(outputs)
                
                batch_size = features.size(0)
                for sample_idx in range(batch_size):
                    sample_id = batch_idx * self.test_loader.batch_size + sample_idx
                    
                    sample_pred = {
                        'sample_id': sample_id,
                        'batch_id': batch_idx,
                        'sample_in_batch': sample_idx
                    }
                    
                    for task in ['cd1', 'cd2', 'cd3', 'va']:
                        if task in wave_predictions:
                            task_data = wave_predictions[task]
                            
                            sample_pred.update({
                                f'{task}_predicted_class': task_data['predicted_classes'][sample_idx],
                                f'{task}_class_confidence': task_data['class_confidences'][sample_idx],
                                f'{task}_wave_direction': task_data['wave_directions'][sample_idx],
                                f'{task}_wave_direction_name': self.wave_analyzer.wave_types[task_data['wave_directions'][sample_idx]],
                                f'{task}_wave_confidence': task_data['wave_confidences'][sample_idx]
                            })
                    
                    if isinstance(labels, list) and len(labels) > sample_idx:
                        label_dict = labels[sample_idx]
                        for task in ['cd1', 'cd2', 'cd3', 'va']:
                            if f'{task}_label' in label_dict:
                                sample_pred[f'{task}_true_label'] = label_dict[f'{task}_label']
                    
                    all_predictions.append(sample_pred)
        
        return pd.DataFrame(all_predictions)
    
    def generate_wave_summary(self, predictions_df):
        """
        Generate wave direction summary statistics
        
        Args:
            predictions_df: Predictions DataFrame
            
        Returns:
            Dictionary with wave summary
        """
        summary = {}
        
        for task in ['cd1', 'cd2', 'cd3', 'va']:
            wave_col = f'{task}_wave_direction'
            conf_col = f'{task}_wave_confidence'
            
            if wave_col in predictions_df.columns:
                task_summary = {
                    'total_predictions': len(predictions_df),
                    'upward_waves': (predictions_df[wave_col] == 1).sum(),
                    'downward_waves': (predictions_df[wave_col] == 0).sum(),
                    'upward_percentage': (predictions_df[wave_col] == 1).mean() * 100,
                    'downward_percentage': (predictions_df[wave_col] == 0).mean() * 100,
                    'average_confidence': predictions_df[conf_col].mean(),
                    'high_confidence_predictions': (predictions_df[conf_col] > 0.8).sum(),
                    'low_confidence_predictions': (predictions_df[conf_col] < 0.5).sum()
                }
                summary[task] = task_summary
        
        return summary

def generate_prediction_analysis(model, test_loader, model_name, device='cpu', output_dir='predictions'):
    """
    Generate comprehensive prediction analysis
    
    Args:
        model: Trained model
        test_loader: Test data loader
        model_name: Name of the model
        device: Computing device
        output_dir: Output directory for results
        
    Returns:
        Tuple of (predictions_df, wave_summary, file_paths)
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"🔮 Generating predictions for {model_name}...")
    generator = PredictionSequenceGenerator(model, test_loader, device)
    predictions_df = generator.generate_predictions()
    
    wave_summary = generator.generate_wave_summary(predictions_df)
    
    pred_file = os.path.join(output_dir, f'{model_name}_predictions_{timestamp}.csv')
    predictions_df.to_csv(pred_file, index=False)
    
    summary_file = os.path.join(output_dir, f'{model_name}_wave_summary_{timestamp}.csv')
    summary_df = pd.DataFrame(wave_summary).T
    summary_df.to_csv(summary_file)
    
    report_file = os.path.join(output_dir, f'{model_name}_wave_report_{timestamp}.md')
    generate_wave_report(predictions_df, wave_summary, model_name, report_file)
    
    print(f"#  Prediction analysis completed for {model_name}")
    print(f"📁 Files saved: {pred_file}, {summary_file}, {report_file}")
    
    return predictions_df, wave_summary, {
        'predictions': pred_file,
        'summary': summary_file,
        'report': report_file
    }

def generate_wave_report(predictions_df, wave_summary, model_name, output_file):
    """
    Generate detailed wave analysis report
    
    Args:
        predictions_df: Predictions DataFrame
        wave_summary: Wave summary statistics
        model_name: Name of the model
        output_file: Output file path
    """
    report_content = f"""# Wave Direction Analysis Report - {model_name}

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


- **Total Predictions**: {len(predictions_df):,}
- **Model**: {model_name}
- **Tasks Analyzed**: CD1, CD2, CD3, VA


"""
    
    for task, summary in wave_summary.items():
        report_content += f"""### {task.upper()} Task

- **Upward Waves**: {summary['upward_waves']:,} ({summary['upward_percentage']:.1f}%)
- **Downward Waves**: {summary['downward_waves']:,} ({summary['downward_percentage']:.1f}%)
- **Average Confidence**: {summary['average_confidence']:.3f}
- **High Confidence (>0.8)**: {summary['high_confidence_predictions']:,}
- **Low Confidence (<0.5)**: {summary['low_confidence_predictions']:,}

"""
    
    report_content += """## 🎯 Confidence Analysis

"""
    
    for task in ['cd1', 'cd2', 'cd3', 'va']:
        conf_col = f'{task}_wave_confidence'
        if conf_col in predictions_df.columns:
            conf_stats = predictions_df[conf_col].describe()
            report_content += f"""### {task.upper()} Confidence Distribution

- **Mean**: {conf_stats['mean']:.3f}
- **Std**: {conf_stats['std']:.3f}
- **Min**: {conf_stats['min']:.3f}
- **Max**: {conf_stats['max']:.3f}
- **25th Percentile**: {conf_stats['25%']:.3f}
- **Median**: {conf_stats['50%']:.3f}
- **75th Percentile**: {conf_stats['75%']:.3f}

"""
    
    report_content += """## 📈 Wave Direction Patterns

"""
    
    for task in ['cd1', 'cd2', 'cd3', 'va']:
        wave_col = f'{task}_wave_direction'
        conf_col = f'{task}_wave_confidence'
        
        if wave_col in predictions_df.columns:
            upward_high_conf = ((predictions_df[wave_col] == 1) & (predictions_df[conf_col] > 0.8)).sum()
            downward_high_conf = ((predictions_df[wave_col] == 0) & (predictions_df[conf_col] > 0.8)).sum()
            
            report_content += f"""### {task.upper()} Patterns

- **High-Confidence Upward Waves**: {upward_high_conf:,}
- **High-Confidence Downward Waves**: {downward_high_conf:,}
- **Confidence Bias**: {'Upward' if upward_high_conf > downward_high_conf else 'Downward' if downward_high_conf > upward_high_conf else 'Balanced'}

"""
    
    report_content += """## 🔍 Interpretation


"""
    
    total_upward = sum(summary['upward_waves'] for summary in wave_summary.values())
    total_downward = sum(summary['downward_waves'] for summary in wave_summary.values())
    total_predictions = total_upward + total_downward
    
    if total_predictions > 0:
        upward_pct = (total_upward / total_predictions) * 100
        downward_pct = (total_downward / total_predictions) * 100
        
        report_content += f"""- **Overall Trend**: {upward_pct:.1f}% Upward, {downward_pct:.1f}% Downward
- **Market Sentiment**: {'Bullish' if upward_pct > 60 else 'Bearish' if downward_pct > 60 else 'Neutral'}
"""
    
    avg_confidence = np.mean([summary['average_confidence'] for summary in wave_summary.values()])
    report_content += f"""

- **Overall Average Confidence**: {avg_confidence:.3f}
- **Model Reliability**: {'High' if avg_confidence > 0.8 else 'Medium' if avg_confidence > 0.6 else 'Low'}
- **Prediction Quality**: {'Excellent' if avg_confidence > 0.85 else 'Good' if avg_confidence > 0.7 else 'Fair' if avg_confidence > 0.5 else 'Poor'}


"""
    
    if avg_confidence > 0.8:
        report_content += "- #  Model shows high confidence in predictions\n"
        report_content += "- #  Wave direction classifications are reliable\n"
    elif avg_confidence > 0.6:
        report_content += "- ⚠️ Model shows moderate confidence\n"
        report_content += "- ⚠️ Consider additional validation for critical decisions\n"
    else:
        report_content += "- #  Model shows low confidence\n"
        report_content += "- #  Predictions should be used with caution\n"
    
    report_content += f"""

- **Softmax Confidence**: Used for class probability calculation
- **Wave Direction Logic**: Upward (1) = Long Fish, Downward (0) = Short Fish
- **Confidence Threshold**: High (>0.8), Medium (0.5-0.8), Low (<0.5)
- **Analysis Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---
*Generated by Wave MAA Prediction Analyzer*
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

if __name__ == "__main__":
    print("🔮 Testing Prediction Analyzer...")
    
    dummy_outputs = {
        'cd1': torch.randn(10, 2),  # 10 samples, 2 classes
        'cd2': torch.randn(10, 2),
        'cd3': torch.randn(10, 2),
        'va': torch.randn(10, 2)
    }
    
    analyzer = WaveDirectionAnalyzer()
    wave_predictions = analyzer.generate_wave_predictions(dummy_outputs)
    
    print("#  Wave direction analysis completed!")
    print(f"#  Generated predictions for {len(wave_predictions)} tasks")
    
    sample_prices = [100, 102, 105, 103, 108]  # Upward trend
    wave_analysis = analyzer.analyze_wave_direction(sample_prices)
    print(f"🌊 Sample wave analysis: {wave_analysis['direction_name']} (confidence: {wave_analysis['confidence_score']:.3f})")

