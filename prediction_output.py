import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from models import SuperMAA, GAN, SimpleGRU, SimpleLSTM, SimpleTransformer
from trainer import SuperMAATrainer
from data_loader import load_wave_maa_data, create_wave_maa_data_loaders
from utils import setup_device, set_random_seed


class PredictionOutputGenerator:
    
    def __init__(self, device, output_dir="prediction_outputs"):
        self.device = device
        self.output_dir = output_dir
        self.models = {}
        self.predictions = {}
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.model_configs = {
            'MAA': {
                'class': SuperMAA,
                'params': {'input_size': 4, 'hidden_size': 256, 'distillation_weight': 0.5, 'cross_supervision_weight': 0.3}
            },
            'GRU': {
                'class': SimpleGRU,
                'params': {'input_size': 4, 'hidden_size': 256}
            },
            'LSTM': {
                'class': SimpleLSTM,
                'params': {'input_size': 4, 'hidden_size': 256}
            },
            'TRANSFORMER': {
                'class': SimpleTransformer,
                'params': {'input_size': 4, 'hidden_size': 256}
            },
            'GAN': {
                'class': GAN,
                'params': {'input_size': 4, 'hidden_size': 256}
            }
        }
        
        self.tasks = ['cd1', 'cd2', 'cd3', 'va']
        
    def load_or_train_model(self, model_name, train_loader, val_loader, epochs=5):
        model_path = f"saved_models/{model_name}_model.pth"
        
        config = self.model_configs[model_name]
        model = config['class'](**config['params']).to(self.device)
        
        if os.path.exists(model_path):
            print(f"Loading existing {model_name} model from {model_path}")
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Successfully loaded {model_name} model")
                return model
            except Exception as e:
                print(f"Failed to load {model_name} model: {e}")
                print(f"Training new {model_name} model...")
        else:
            print(f"No existing {model_name} model found. Training new model...")
        
        trainer = SuperMAATrainer(model, self.device, model_name=model_name)
        trained_model, history = trainer.train(train_loader, val_loader, epochs)
        
        os.makedirs("saved_models", exist_ok=True)
        torch.save(trained_model.state_dict(), model_path)
        print(f"Saved {model_name} model to {model_path}")
        
        return trained_model
    
    def generate_predictions(self, model, model_name, test_loader):
        model.eval()
        
        all_predictions = {
            'sample_indices': [],
            'features': [],
            'true_labels': {},
            'true_lengths': {},
            'predicted_labels': {},
            'predicted_lengths': {},
            'prediction_probabilities': {}
        }
        
        for task in self.tasks:
            all_predictions['true_labels'][task] = []
            all_predictions['true_lengths'][task] = []
            all_predictions['predicted_labels'][task] = []
            all_predictions['predicted_lengths'][task] = []
            all_predictions['prediction_probabilities'][task] = []
        
        print(f"Generating predictions for {model_name}...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                features = batch['features'].to(self.device)
                batch_size = features.size(0)
                
                outputs = model(features)
                
                all_predictions['features'].extend(features.cpu().numpy().tolist())
                all_predictions['sample_indices'].extend(
                    list(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
                )
                
                for task in self.tasks:
                    if task in outputs['tasks']:
                        true_labels = batch[f'{task}_label'].cpu().numpy()
                        true_lengths = batch[f'{task}_length'].cpu().numpy()
                        
                        task_logits = outputs['tasks'][task].cpu()
                        task_probs = torch.softmax(task_logits, dim=1).numpy()
                        predicted_labels = torch.argmax(task_logits, dim=1).numpy()
                        
                        if f'{task}_length' in outputs['lengths']:
                            predicted_lengths = outputs['lengths'][f'{task}_length'].cpu().squeeze().numpy()
                        else:
                            predicted_lengths = np.zeros_like(true_lengths)
                        
                        all_predictions['true_labels'][task].extend(true_labels.tolist())
                        all_predictions['true_lengths'][task].extend(true_lengths.tolist())
                        all_predictions['predicted_labels'][task].extend(predicted_labels.tolist())
                        all_predictions['predicted_lengths'][task].extend(predicted_lengths.tolist())
                        all_predictions['prediction_probabilities'][task].extend(task_probs.tolist())
        
        return all_predictions
    
    def save_predictions_to_csv(self, predictions, model_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        prediction_data = []
        
        num_samples = len(predictions['sample_indices'])
        
        for i in range(num_samples):
            base_row = {
                'Sample_Index': predictions['sample_indices'][i],
                'Model': model_name,
            }
            
            if len(predictions['features'][i]) > 0:
                feature_seq = predictions['features'][i]
                if len(feature_seq) > 0 and len(feature_seq[0]) >= 4:
                    base_row.update({
                        'Feature_Open': feature_seq[0][0],
                        'Feature_High': feature_seq[0][1],
                        'Feature_Low': feature_seq[0][2],
                        'Feature_Close': feature_seq[0][3]
                    })
            
            for task in self.tasks:
                if task in predictions['true_labels'] and i < len(predictions['true_labels'][task]):
                    task_row = base_row.copy()
                    task_row.update({
                        'Task': task.upper(),
                        'True_Label': predictions['true_labels'][task][i],
                        'Predicted_Label': predictions['predicted_labels'][task][i],
                        'True_Length': predictions['true_lengths'][task][i],
                        'Predicted_Length': predictions['predicted_lengths'][task][i],
                        'Prediction_Correct': int(predictions['true_labels'][task][i] == predictions['predicted_labels'][task][i]),
                        'Length_Error': abs(predictions['true_lengths'][task][i] - predictions['predicted_lengths'][task][i])
                    })
                    
                    if i < len(predictions['prediction_probabilities'][task]):
                        probs = predictions['prediction_probabilities'][task][i]
                        task_row.update({
                            'Prob_Class_0': probs[0] if len(probs) > 0 else 0.0,
                            'Prob_Class_1': probs[1] if len(probs) > 1 else 0.0,
                            'Confidence': max(probs) if len(probs) > 0 else 0.0
                        })
                    
                    prediction_data.append(task_row)
        
        df = pd.DataFrame(prediction_data)
        csv_filename = f"{self.output_dir}/{model_name}_predictions_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Saved {model_name} predictions to {csv_filename}")
        
        return csv_filename
    
    def save_predictions_to_json(self, predictions, model_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_predictions = {}
        for key, value in predictions.items():
            if isinstance(value, dict):
                json_predictions[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        json_predictions[key][sub_key] = sub_value.tolist()
                    elif isinstance(sub_value, list):
                        json_predictions[key][sub_key] = [
                            float(x) if isinstance(x, (np.float32, np.float64)) else x 
                            for x in sub_value
                        ]
                    else:
                        json_predictions[key][sub_key] = sub_value
            elif isinstance(value, np.ndarray):
                json_predictions[key] = value.tolist()
            elif isinstance(value, list):
                json_predictions[key] = [
                    float(x) if isinstance(x, (np.float32, np.float64)) else x 
                    for x in value
                ]
            else:
                json_predictions[key] = value
        
        json_predictions['metadata'] = {
            'model_name': model_name,
            'timestamp': timestamp,
            'num_samples': len(predictions['sample_indices']),
            'tasks': self.tasks,
            'description': f'Detailed predictions for {model_name} model on test set'
        }
        
        json_filename = f"{self.output_dir}/{model_name}_predictions_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(json_predictions, f, indent=2)
        
        print(f"Saved {model_name} detailed predictions to {json_filename}")
        return json_filename
    
    def generate_summary_report(self, all_predictions):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary_data = []
        
        for model_name, predictions in all_predictions.items():
            for task in self.tasks:
                if task in predictions['true_labels']:
                    true_labels = np.array(predictions['true_labels'][task])
                    pred_labels = np.array(predictions['predicted_labels'][task])
                    true_lengths = np.array(predictions['true_lengths'][task])
                    pred_lengths = np.array(predictions['predicted_lengths'][task])
                    
                    accuracy = np.mean(true_labels == pred_labels)
                    length_mae = np.mean(np.abs(true_lengths - pred_lengths))
                    
                    class_0_mask = true_labels == 0
                    class_1_mask = true_labels == 1
                    
                    s_accuracy = np.mean(pred_labels[class_0_mask] == 0) if np.any(class_0_mask) else 0.0
                    l_accuracy = np.mean(pred_labels[class_1_mask] == 1) if np.any(class_1_mask) else 0.0
                    
                    summary_data.append({
                        'Model': model_name,
                        'Task': task.upper(),
                        'Total_Samples': len(true_labels),
                        'Accuracy': accuracy,
                        'S_Accuracy': s_accuracy,
                        'L_Accuracy': l_accuracy,
                        'Length_MAE': length_mae,
                        'Class_0_Count': np.sum(class_0_mask),
                        'Class_1_Count': np.sum(class_1_mask)
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = f"{self.output_dir}/prediction_summary_{timestamp}.csv"
        summary_df.to_csv(summary_filename, index=False)
        print(f"Saved prediction summary to {summary_filename}")
        
        return summary_filename
    
    def run_prediction_output(self, data_path="cvda_dataset", epochs=5):
        print("=" * 80)
        print("Wave MAA Framework - Prediction Output Generation")
        print("=" * 80)
        
        print("Loading data...")
        data = load_wave_maa_data(data_path)
        train_loader, val_loader, test_loader = create_wave_maa_data_loaders(data, batch_size=32)
        
        print(f"Data loaded successfully:")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
        print(f"  Test samples: {len(test_loader.dataset)}")
        
        all_predictions = {}
        
        for model_name in self.model_configs.keys():
            print(f"\n{'-' * 60}")
            print(f"Processing {model_name} model...")
            print(f"{'-' * 60}")
            
            model = self.load_or_train_model(model_name, train_loader, val_loader, epochs)
            
            predictions = self.generate_predictions(model, model_name, test_loader)
            all_predictions[model_name] = predictions
            
            csv_file = self.save_predictions_to_csv(predictions, model_name)
            json_file = self.save_predictions_to_json(predictions, model_name)
            
            print(f"Completed {model_name} prediction output")
        
        print(f"\n{'-' * 60}")
        print("Generating summary report...")
        print(f"{'-' * 60}")
        summary_file = self.generate_summary_report(all_predictions)
        
        print(f"\n{'=' * 80}")
        print("Prediction Output Generation Completed!")
        print(f"{'=' * 80}")
        print(f"Output directory: {self.output_dir}")
        print(f"Summary report: {summary_file}")
        print(f"Total models processed: {len(self.model_configs)}")
        print(f"Total tasks per model: {len(self.tasks)}")
        
        return all_predictions


def main():
    set_random_seed(42)
    
    device = setup_device()
    print(f"Using device: {device}")
    
    generator = PredictionOutputGenerator(device)
    
    predictions = generator.run_prediction_output(epochs=3)
    
    print("\nPrediction output generation completed successfully!")


if __name__ == "__main__":
    main()

