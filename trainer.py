import torch
import torch.nn as nn
from tqdm import tqdm
import copy

class SuperMAATrainer:
    def __init__(self, model, device, learning_rate=1e-4, model_name="SuperMAA"):
        self.model = model
        self.device = device
        self.model_name = model_name
        
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        self.adversarial_loss = nn.BCELoss()
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'distillation_loss': [],
            'cross_supervision_loss': []
        }
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_main_loss = 0
        total_distill_loss = 0
        total_cross_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training {self.model_name}")
        
        for batch in progress_bar:
            features = batch['features'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if hasattr(self.model, 'compute_total_loss'):
                outputs = self.model(features, compute_losses=True)
                
                targets = {}
                available_tasks = list(outputs['tasks'].keys())
                
                for task in available_tasks:
                    targets[f'{task}_label'] = batch[f'{task}_label'].to(self.device)
                    targets[f'{task}_length'] = batch[f'{task}_length'].to(self.device)
                
                total_batch_loss, loss_components = self.model.compute_total_loss(outputs, targets)
                
                main_loss = loss_components['main_task_loss']
                distill_loss = loss_components['distillation_loss']
                cross_loss = loss_components['cross_supervision_loss']
                
                total_main_loss += main_loss
                total_distill_loss += distill_loss
                total_cross_loss += cross_loss
                
            else:
                outputs = self.model(features)
                
                total_batch_loss = 0
                available_tasks = list(outputs['tasks'].keys())
                
                for task in available_tasks:
                    task_labels = batch[f'{task}_label'].to(self.device)
                    task_lengths = batch[f'{task}_length'].to(self.device)
                    
                    total_batch_loss += self.classification_loss(outputs['tasks'][task], task_labels)
                    
                    if f'{task}_length' in outputs['lengths']:
                        total_batch_loss += 0.1 * self.regression_loss(
                            outputs['lengths'][f'{task}_length'].squeeze(), task_lengths
                        )
                
                if 'discriminator' in outputs:
                    real_labels = torch.ones(features.size(0), 1).to(self.device)
                    total_batch_loss += 0.1 * self.adversarial_loss(outputs['discriminator'], real_labels)
                
                total_main_loss += total_batch_loss.item()
            
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            
            if 'cd1' in outputs['tasks']:
                cd1_labels = batch['cd1_label'].to(self.device)
                _, predicted = torch.max(outputs['tasks']['cd1'], 1)
                correct_predictions += (predicted == cd1_labels).sum().item()
                total_predictions += cd1_labels.size(0)
            
            current_acc = correct_predictions / total_predictions if total_predictions > 0 else 0
            if hasattr(self.model, 'compute_total_loss'):
                progress_bar.set_postfix({
                    'Total': f'{total_batch_loss.item():.4f}',
                    'Main': f'{main_loss:.4f}',
                    'Distill': f'{distill_loss:.4f}',
                    'Cross': f'{cross_loss:.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
            else:
                progress_bar.set_postfix({
                    'Loss': f'{total_batch_loss.item():.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
        
        self.scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = correct_predictions / total_predictions if total_predictions > 0 else 0
        avg_main_loss = total_main_loss / len(train_loader)
        avg_distill_loss = total_distill_loss / len(train_loader) if total_distill_loss > 0 else 0
        avg_cross_loss = total_cross_loss / len(train_loader) if total_cross_loss > 0 else 0
        
        return avg_loss, avg_acc, avg_main_loss, avg_distill_loss, avg_cross_loss
    
    def validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                
                outputs = self.model(features)
                
                batch_loss = 0
                available_tasks = list(outputs['tasks'].keys())
                
                for task in available_tasks:
                    task_labels = batch[f'{task}_label'].to(self.device)
                    batch_loss += self.classification_loss(outputs['tasks'][task], task_labels)
                
                total_loss += batch_loss.item()
                
                if 'cd1' in outputs['tasks']:
                    cd1_labels = batch['cd1_label'].to(self.device)
                    _, predicted = torch.max(outputs['tasks']['cd1'], 1)
                    correct_predictions += (predicted == cd1_labels).sum().item()
                    total_predictions += cd1_labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return avg_loss, avg_acc
    
    def train(self, train_loader, val_loader, epochs=5):
        print(f"Training {self.model_name} model...")
        print(f"   Epochs: {epochs}")
        print(f"   Training samples: {len(train_loader.dataset)}")
        print(f"   Validation samples: {len(val_loader.dataset)}")
        
        has_enhanced_features = hasattr(self.model, 'compute_total_loss')
        if has_enhanced_features:
            print(f"   Enhanced features detected: Distillation + Cross-supervision")
        else:
            print(f"   Standard training mode")
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 60)
            
            train_results = self.train_epoch(train_loader)
            train_loss, train_acc = train_results[0], train_results[1]
            
            if has_enhanced_features:
                train_main_loss, train_distill_loss, train_cross_loss = train_results[2], train_results[3], train_results[4]
            
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(self.model.state_dict())
                print(f"New best model saved! Validation accuracy: {val_acc:.4f}")
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if has_enhanced_features:
                self.history['distillation_loss'].append(train_distill_loss)
                self.history['cross_supervision_loss'].append(train_cross_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            if has_enhanced_features:
                print(f"  Main Task Loss: {train_main_loss:.4f}")
                print(f"  Distillation Loss: {train_distill_loss:.4f}")
                print(f"  Cross-supervision Loss: {train_cross_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f"\n{self.model_name} training completed!")
        print(f"   Best validation accuracy: {best_val_acc:.4f}")
        
        if has_enhanced_features:
            final_distill = self.history['distillation_loss'][-1] if self.history['distillation_loss'] else 0
            final_cross = self.history['cross_supervision_loss'][-1] if self.history['cross_supervision_loss'] else 0
            print(f"   Final distillation loss: {final_distill:.4f}")
            print(f"   Final cross-supervision loss: {final_cross:.4f}")
        
        return self.model, self.history

