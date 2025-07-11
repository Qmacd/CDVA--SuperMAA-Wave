import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json


class SequenceAnalyzer:
    
    def __init__(self, output_dir="sequence_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def load_prediction_data(self, prediction_dir="prediction_outputs"):
        prediction_files = {}
        
        if not os.path.exists(prediction_dir):
            print(f"Prediction directory {prediction_dir} not found!")
            return prediction_files
        
        for filename in os.listdir(prediction_dir):
            if filename.endswith('_predictions.csv'):
                model_name = filename.split('_predictions')[0]
                filepath = os.path.join(prediction_dir, filename)
                try:
                    df = pd.read_csv(filepath)
                    prediction_files[model_name] = df
                    print(f"Loaded predictions for {model_name}: {len(df)} records")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return prediction_files
    
    def analyze_prediction_patterns(self, predictions_dict):
        analysis_results = {}
        
        for model_name, df in predictions_dict.items():
            model_analysis = {}
            
            for task in df['Task'].unique():
                task_df = df[df['Task'] == task].copy()
                
                if len(task_df) == 0:
                    continue
                
                task_analysis = {
                    'total_samples': len(task_df),
                    'accuracy': task_df['Prediction_Correct'].mean(),
                    'avg_confidence': task_df['Confidence'].mean() if 'Confidence' in task_df.columns else 0,
                    'length_mae': task_df['Length_Error'].mean() if 'Length_Error' in task_df.columns else 0
                }
                
                class_dist = task_df['True_Label'].value_counts().to_dict()
                task_analysis['class_distribution'] = class_dist
                
                pred_dist = task_df['Predicted_Label'].value_counts().to_dict()
                task_analysis['prediction_distribution'] = pred_dist
                
                confusion_matrix = pd.crosstab(
                    task_df['True_Label'], 
                    task_df['Predicted_Label'], 
                    margins=True
                )
                task_analysis['confusion_matrix'] = confusion_matrix.to_dict()
                
                for class_label in [0, 1]:
                    class_mask = task_df['True_Label'] == class_label
                    if class_mask.sum() > 0:
                        class_data = task_df[class_mask]
                        task_analysis[f'class_{class_label}_accuracy'] = class_data['Prediction_Correct'].mean()
                        task_analysis[f'class_{class_label}_confidence'] = class_data['Confidence'].mean() if 'Confidence' in class_data.columns else 0
                        task_analysis[f'class_{class_label}_length_error'] = class_data['Length_Error'].mean() if 'Length_Error' in class_data.columns else 0
                
                if 'Confidence' in task_df.columns:
                    confidence_bins = pd.cut(task_df['Confidence'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                    confidence_accuracy = task_df.groupby(confidence_bins)['Prediction_Correct'].mean().to_dict()
                    task_analysis['confidence_vs_accuracy'] = confidence_accuracy
                
                model_analysis[task] = task_analysis
            
            analysis_results[model_name] = model_analysis
        
        return analysis_results
    
    def generate_sequence_comparison(self, predictions_dict):
        comparison_results = {}
        
        all_tasks = set()
        for df in predictions_dict.values():
            all_tasks.update(df['Task'].unique())
        
        for task in all_tasks:
            task_comparison = {}
            
            task_predictions = {}
            for model_name, df in predictions_dict.items():
                task_df = df[df['Task'] == task].copy()
                if len(task_df) > 0:
                    task_df = task_df.sort_values('Sample_Index')
                    task_predictions[model_name] = task_df
            
            if len(task_predictions) < 2:
                continue
            
            common_indices = None
            for model_name, task_df in task_predictions.items():
                indices = set(task_df['Sample_Index'].values)
                if common_indices is None:
                    common_indices = indices
                else:
                    common_indices = common_indices.intersection(indices)
            
            if len(common_indices) == 0:
                continue
            
            common_indices = sorted(list(common_indices))
            
            agreement_data = []
            for idx in common_indices:
                sample_data = {'Sample_Index': idx}
                predictions = {}
                true_label = None
                
                for model_name, task_df in task_predictions.items():
                    sample_row = task_df[task_df['Sample_Index'] == idx]
                    if len(sample_row) > 0:
                        sample_row = sample_row.iloc[0]
                        predictions[model_name] = sample_row['Predicted_Label']
                        if true_label is None:
                            true_label = sample_row['True_Label']
                        sample_data[f'{model_name}_pred'] = sample_row['Predicted_Label']
                        sample_data[f'{model_name}_conf'] = sample_row.get('Confidence', 0)
                
                sample_data['True_Label'] = true_label
                
                pred_values = list(predictions.values())
                sample_data['full_agreement'] = len(set(pred_values)) == 1
                sample_data['majority_prediction'] = max(set(pred_values), key=pred_values.count)
                sample_data['agreement_count'] = pred_values.count(sample_data['majority_prediction'])
                sample_data['correct_majority'] = sample_data['majority_prediction'] == true_label
                
                agreement_data.append(sample_data)
            
            agreement_df = pd.DataFrame(agreement_data)
            task_comparison['total_common_samples'] = len(agreement_df)
            task_comparison['full_agreement_rate'] = agreement_df['full_agreement'].mean()
            task_comparison['majority_accuracy'] = agreement_df['correct_majority'].mean()
            
            model_names = list(task_predictions.keys())
            pair_agreements = {}
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names[i+1:], i+1):
                    if f'{model1}_pred' in agreement_df.columns and f'{model2}_pred' in agreement_df.columns:
                        agreement_rate = (agreement_df[f'{model1}_pred'] == agreement_df[f'{model2}_pred']).mean()
                        pair_agreements[f'{model1}_vs_{model2}'] = agreement_rate
            
            task_comparison['pairwise_agreements'] = pair_agreements
            task_comparison['agreement_details'] = agreement_df.to_dict('records')
            
            comparison_results[task] = task_comparison
        
        return comparison_results
    
    def generate_error_analysis(self, predictions_dict):
        error_analysis = {}
        
        for model_name, df in predictions_dict.items():
            model_errors = {}
            
            for task in df['Task'].unique():
                task_df = df[df['Task'] == task].copy()
                error_df = task_df[task_df['Prediction_Correct'] == 0].copy()
                
                if len(error_df) == 0:
                    model_errors[task] = {'error_count': 0}
                    continue
                
                task_errors = {
                    'error_count': len(error_df),
                    'error_rate': len(error_df) / len(task_df),
                    'total_samples': len(task_df)
                }
                
                false_positives = error_df[error_df['True_Label'] == 0]
                false_negatives = error_df[error_df['True_Label'] == 1]
                
                task_errors['false_positive_count'] = len(false_positives)
                task_errors['false_negative_count'] = len(false_negatives)
                task_errors['false_positive_rate'] = len(false_positives) / len(task_df[task_df['True_Label'] == 0]) if len(task_df[task_df['True_Label'] == 0]) > 0 else 0
                task_errors['false_negative_rate'] = len(false_negatives) / len(task_df[task_df['True_Label'] == 1]) if len(task_df[task_df['True_Label'] == 1]) > 0 else 0
                
                if 'Confidence' in error_df.columns:
                    task_errors['avg_error_confidence'] = error_df['Confidence'].mean()
                    task_errors['low_confidence_errors'] = (error_df['Confidence'] < 0.6).sum()
                    task_errors['high_confidence_errors'] = (error_df['Confidence'] > 0.8).sum()
                
                if 'Length_Error' in error_df.columns:
                    task_errors['avg_length_error'] = error_df['Length_Error'].mean()
                    task_errors['max_length_error'] = error_df['Length_Error'].max()
                    task_errors['length_error_std'] = error_df['Length_Error'].std()
                
                model_errors[task] = task_errors
            
            error_analysis[model_name] = model_errors
        
        return error_analysis
    
    def save_analysis_results(self, analysis_results, comparison_results, error_analysis):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        pattern_file = f"{self.output_dir}/pattern_analysis_{timestamp}.json"
        with open(pattern_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        print(f"Saved pattern analysis to {pattern_file}")
        
        comparison_file = f"{self.output_dir}/model_comparison_{timestamp}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        print(f"Saved model comparison to {comparison_file}")
        
        error_file = f"{self.output_dir}/error_analysis_{timestamp}.json"
        with open(error_file, 'w') as f:
            json.dump(error_analysis, f, indent=2, default=str)
        print(f"Saved error analysis to {error_file}")
        
        summary_data = []
        for model_name, model_analysis in analysis_results.items():
            for task, task_analysis in model_analysis.items():
                summary_data.append({
                    'Model': model_name,
                    'Task': task,
                    'Total_Samples': task_analysis.get('total_samples', 0),
                    'Accuracy': task_analysis.get('accuracy', 0),
                    'Avg_Confidence': task_analysis.get('avg_confidence', 0),
                    'Length_MAE': task_analysis.get('length_mae', 0),
                    'Class_0_Accuracy': task_analysis.get('class_0_accuracy', 0),
                    'Class_1_Accuracy': task_analysis.get('class_1_accuracy', 0)
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv = f"{self.output_dir}/analysis_summary_{timestamp}.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"Saved analysis summary to {summary_csv}")
        
        return pattern_file, comparison_file, error_file, summary_csv
    
    def run_full_analysis(self, prediction_dir="prediction_outputs"):
        print("=" * 80)
        print("Wave MAA Framework - Sequence Analysis")
        print("=" * 80)
        
        print("Loading prediction data...")
        predictions_dict = self.load_prediction_data(prediction_dir)
        
        if not predictions_dict:
            print("No prediction data found!")
            return
        
        print(f"Loaded predictions for {len(predictions_dict)} models")
        
        print("\nAnalyzing prediction patterns...")
        analysis_results = self.analyze_prediction_patterns(predictions_dict)
        
        print("Comparing model predictions...")
        comparison_results = self.generate_sequence_comparison(predictions_dict)
        
        print("Analyzing prediction errors...")
        error_analysis = self.generate_error_analysis(predictions_dict)
        
        print("\nSaving analysis results...")
        files = self.save_analysis_results(analysis_results, comparison_results, error_analysis)
        
        print(f"\n{'=' * 80}")
        print("Sequence Analysis Completed!")
        print(f"{'=' * 80}")
        print(f"Output directory: {self.output_dir}")
        print(f"Generated files: {len(files)}")
        
        return analysis_results, comparison_results, error_analysis


def main():
    analyzer = SequenceAnalyzer()
    results = analyzer.run_full_analysis()
    print("\nSequence analysis completed successfully!")


if __name__ == "__main__":
    main()

