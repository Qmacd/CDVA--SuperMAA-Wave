"""
Test script for Enhanced MAA with distillation and cross-supervision
"""

import torch
import torch.nn as nn
from models import SuperMAA, SimpleGRU, SimpleLSTM, SimpleTransformer, GAN
from data_loader import WaveMAA_Dataset
from trainer import SuperMAATrainer
import warnings
warnings.filterwarnings('ignore')

def test_enhanced_maa():
    """Test the enhanced MAA model with distillation features"""
    print("Testing Enhanced MAA Model")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    batch_size = 16
    seq_len = 20
    feature_dim = 4
    num_samples = 100
    
    print(f"Creating demo data: {num_samples} samples, {seq_len} timesteps, {feature_dim} features")
    
    demo_features = torch.randn(num_samples, seq_len, feature_dim)
    demo_labels = {
        'cd1': torch.randint(0, 2, (num_samples,)),
        'cd2': torch.randint(0, 2, (num_samples,)),
        'cd3': torch.randint(0, 2, (num_samples,)),
        'va': torch.randint(0, 2, (num_samples,))
    }
    demo_lengths = {
        'cd1': torch.rand(num_samples) * 5 + 2,
        'cd2': torch.rand(num_samples) * 10 + 5,
        'cd3': torch.rand(num_samples) * 15 + 10,
        'va': torch.rand(num_samples) * 12 + 3
    }
    
    dataset = WaveMAA_Dataset(
        demo_features, demo_labels['cd1'], demo_labels['cd2'], 
        demo_labels['cd3'], demo_labels['va'],
        demo_lengths['cd1'], demo_lengths['cd2'], 
        demo_lengths['cd3'], demo_lengths['va']
    )
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Dataset created: {len(dataset)} samples")
    print(f"Data loader created: {len(data_loader)} batches")
    
    print("\n1. Testing Enhanced SuperMAA Model")
    print("-" * 30)
    
    enhanced_maa = SuperMAA(
        input_size=feature_dim, 
        hidden_size=128,  # Smaller for testing
        available_tasks=['cd1', 'cd2', 'cd3', 'va'],
        distillation_weight=0.5,
        cross_supervision_weight=0.3,
        temperature=4.0
    ).to(device)
    
    print(f"Enhanced MAA created with {sum(p.numel() for p in enhanced_maa.parameters())} parameters")
    
    sample_batch = next(iter(data_loader))
    features = sample_batch['features'].to(device)
    
    print(f"Testing forward pass with batch shape: {features.shape}")
    
    with torch.no_grad():
        outputs = enhanced_maa(features)
        print("#  Standard forward pass successful")
        print(f"   Tasks output keys: {list(outputs['tasks'].keys())}")
        print(f"   Lengths output keys: {list(outputs['lengths'].keys())}")
        print(f"   Additional outputs: {[k for k in outputs.keys() if k not in ['tasks', 'lengths']]}")
    
    with torch.no_grad():
        enhanced_outputs = enhanced_maa(features, compute_losses=True)
        print("#  Enhanced forward pass successful")
        print(f"   Agent features: {list(enhanced_outputs['agent_features'].keys())}")
        print(f"   Distilled features: {list(enhanced_outputs['distilled_features'].keys())}")
        print(f"   Has consistency scores: {'consistency_scores' in enhanced_outputs}")
        print(f"   Has cross attention: {'cross_attention' in enhanced_outputs}")
    
    print("\n2. Testing Enhanced Loss Computation")
    print("-" * 30)
    
    targets = {}
    for task in ['cd1', 'cd2', 'cd3', 'va']:
        targets[f'{task}_label'] = sample_batch[f'{task}_label'].to(device)
        targets[f'{task}_length'] = sample_batch[f'{task}_length'].to(device)
    
    total_loss, loss_components = enhanced_maa.compute_total_loss(enhanced_outputs, targets)
    
    print("#  Enhanced loss computation successful")
    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   Main task loss: {loss_components['main_task_loss']:.4f}")
    print(f"   Distillation loss: {loss_components['distillation_loss']:.4f}")
    print(f"   Cross-supervision loss: {loss_components['cross_supervision_loss']:.4f}")
    print(f"   Task losses: {loss_components['task_losses']}")
    print(f"   Distillation details: {loss_components['distill_details']}")
    
    print("\n3. Testing Enhanced Trainer")
    print("-" * 30)
    
    trainer = SuperMAATrainer(enhanced_maa, device, learning_rate=1e-3, model_name="Enhanced_SuperMAA")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    
    print("Testing one training epoch...")
    train_results = trainer.train_epoch(train_loader)
    print(f"#  Training epoch successful")
    print(f"   Train loss: {train_results[0]:.4f}")
    print(f"   Train accuracy: {train_results[1]:.4f}")
    if len(train_results) > 2:
        print(f"   Main loss: {train_results[2]:.4f}")
        print(f"   Distillation loss: {train_results[3]:.4f}")
        print(f"   Cross-supervision loss: {train_results[4]:.4f}")
    
    print("Testing validation...")
    val_loss, val_acc = trainer.validate_epoch(val_loader)
    print(f"#  Validation successful")
    print(f"   Val loss: {val_loss:.4f}")
    print(f"   Val accuracy: {val_acc:.4f}")
    
    print("\n4. Testing Baseline Models")
    print("-" * 30)
    
    baseline_models = {
        'SimpleGRU': SimpleGRU(feature_dim, 128, ['cd1', 'cd2', 'cd3', 'va']),
        'SimpleLSTM': SimpleLSTM(feature_dim, 128, ['cd1', 'cd2', 'cd3', 'va']),
        'SimpleTransformer': SimpleTransformer(feature_dim, 128, ['cd1', 'cd2', 'cd3', 'va']),
        'GAN': GAN(feature_dim, 128, ['cd1', 'cd2', 'cd3', 'va'])
    }
    
    for name, model in baseline_models.items():
        model = model.to(device)
        with torch.no_grad():
            outputs = model(features)
            print(f"#  {name} forward pass successful")
            print(f"   Tasks: {list(outputs['tasks'].keys())}")
            print(f"   Lengths: {list(outputs['lengths'].keys())}")
    
    print("\n" + "=" * 50)
    print("#  All tests passed successfully!")
    print("Enhanced MAA with distillation and cross-supervision is working correctly.")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    try:
        test_enhanced_maa()
    except Exception as e:
        print(f"\n#  Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

