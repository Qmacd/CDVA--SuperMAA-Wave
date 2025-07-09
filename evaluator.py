import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error
from tqdm import tqdm

class ModelEvaluator:
    """Model evaluator for generating CSV results"""
    
    def __init__(self, model, device, model_name):
        self.model = model
        self.device = device
        self.model_name = model_name
        
    def evaluate(self, test_loader):
        """
        Evaluate model and return results in CSV format
        
        Returns:
            List of dictionaries with results for each task
        """
        print(f"#  Evaluating {self.model_name} model...")
        
        self.model.eval()
        
        task_predictions = {'cd1': [], 'cd2': [], 'cd3': [], 'va': []}
        task_labels = {'cd1': [], 'cd2': [], 'cd3': [], 'va': []}
        length_predictions = {'cd1': [], 'cd2': [], 'cd3': [], 'va': []}
        length_labels = {'cd1': [], 'cd2': [], 'cd3': [], 'va': []}
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {self.model_name}"):
                features = batch['features'].to(self.device)
                
                cd1_labels = batch['cd1_label'].cpu().numpy()
                cd2_labels = batch['cd2_label'].cpu().numpy()
                cd3_labels = batch['cd3_label'].cpu().numpy()
                va_labels = batch['va_label'].cpu().numpy()
                
                cd1_lengths = batch['cd1_length'].cpu().numpy()
                cd2_lengths = batch['cd2_length'].cpu().numpy()
                cd3_lengths = batch['cd3_length'].cpu().numpy()
                va_lengths = batch['va_length'].cpu().numpy()
                
                outputs = self.model(features)
                
                if 'cd1' in outputs['tasks']:
                    pred = torch.argmax(outputs['tasks']['cd1'], dim=1).cpu().numpy()
                    task_predictions['cd1'].extend(pred)
                    task_labels['cd1'].extend(cd1_labels)
                
                if 'cd2' in outputs['tasks']:
                    pred = torch.argmax(outputs['tasks']['cd2'], dim=1).cpu().numpy()
                    task_predictions['cd2'].extend(pred)
                    task_labels['cd2'].extend(cd2_labels)
                
                if 'cd3' in outputs['tasks']:
                    pred = torch.argmax(outputs['tasks']['cd3'], dim=1).cpu().numpy()
                    task_predictions['cd3'].extend(pred)
                    task_labels['cd3'].extend(cd3_labels)
                
                if 'va' in outputs['tasks']:
                    pred = torch.argmax(outputs['tasks']['va'], dim=1).cpu().numpy()
                    task_predictions['va'].extend(pred)
                    task_labels['va'].extend(va_labels)
                
                if 'cd1_length' in outputs['lengths']:
                    pred = outputs['lengths']['cd1_length'].squeeze().cpu().numpy()
                    length_predictions['cd1'].extend(pred)
                    length_labels['cd1'].extend(cd1_lengths)
                
                if 'cd2_length' in outputs['lengths']:
                    pred = outputs['lengths']['cd2_length'].squeeze().cpu().numpy()
                    length_predictions['cd2'].extend(pred)
                    length_labels['cd2'].extend(cd2_lengths)
                
                if 'cd3_length' in outputs['lengths']:
                    pred = outputs['lengths']['cd3_length'].squeeze().cpu().numpy()
                    length_predictions['cd3'].extend(pred)
                    length_labels['cd3'].extend(cd3_lengths)
                
                if 'va_length' in outputs['lengths']:
                    pred = outputs['lengths']['va_length'].squeeze().cpu().numpy()
                    length_predictions['va'].extend(pred)
                    length_labels['va'].extend(va_lengths)
        
        results = []
        
        for task in ['cd1', 'cd2', 'cd3', 'va']:
            if task_labels[task]:  # If task has data
                acc = accuracy_score(task_labels[task], task_predictions[task])
                f1 = f1_score(task_labels[task], task_predictions[task], average='weighted')
                prec = precision_score(task_labels[task], task_predictions[task], average='weighted')
                rec = recall_score(task_labels[task], task_predictions[task], average='weighted')
                
                s_correct = sum(1 for true_label, pred in zip(task_labels[task], task_predictions[task]) 
                               if true_label == 0 and pred == 0)
                s_total = sum(1 for label in task_labels[task] if label == 0)
                s_acc = s_correct / s_total if s_total > 0 else 0
                
                l_correct = sum(1 for true_label, pred in zip(task_labels[task], task_predictions[task]) 
                               if true_label == 1 and pred == 1)
                l_total = sum(1 for label in task_labels[task] if label == 1)
                l_acc = l_correct / l_total if l_total > 0 else 0
                
                length_mae = 0
                if length_labels[task] and length_predictions[task]:
                    length_mae = mean_absolute_error(length_labels[task], length_predictions[task])
                
                result = {
                    'Model': self.model_name,
                    'Task': task.upper(),
                    'Accuracy': acc,
                    'F1_Score': f1,
                    'Precision': prec,
                    'Recall': rec,
                    'S_Accuracy': s_acc,
                    'L_Accuracy': l_acc,
                    'Length_MAE': length_mae
                }
                
                results.append(result)
                
                print(f"   {task.upper()}: Acc={acc:.4f}, F1={f1:.4f}, S_Acc={s_acc:.4f}, L_Acc={l_acc:.4f}, MAE={length_mae:.4f}")
        
        return results

