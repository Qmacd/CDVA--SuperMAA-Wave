# Enhanced Wave MAA Framework for AutoDL

## Overview
Enhanced Multi-Agent Architecture (MAA) framework with distillation, cross-distillation, and cross-supervision loss for financial time series analysis.

## Dataset Placement
Place the `cvda_dataset` folder in one of these locations:
- `./cvda_dataset/` (recommended)
- `./data/cvda_dataset/`
- `/root/autodl-tmp/cvda_dataset/`
- `/root/wave_maa/cvda_dataset/`
- `/root/cvda_dataset/`

## Quick Start on AutoDL

1. Upload and extract this package
2. Install dependencies:
```bash
pip install torch pandas scikit-learn tqdm matplotlib seaborn
```

3. Run experiment:
```bash
python run_autodl_experiment.py
```

## Project Structure

- `models.py`: Enhanced MAA + baseline models
- `enhanced_trainer.py`: Training with distillation support
- `trainer.py`: Standard trainer for baseline models
- `data_loader.py`: Data loading and preprocessing
- `evaluator.py`: Model evaluation with CSV output
- `plot_generator.py`: Visualization and plotting
- `prediction_analyzer.py`: Prediction analysis tools
- `technical_indicators.py`: Technical indicator calculations
- `utils.py`: Utility functions
- `demo.py`: Demo script
- `run_autodl_experiment.py`: Main experiment script
- `run_experiment.py`: Alternative experiment script
- `run_autodl.sh`: Shell script for AutoDL
- `requirements.txt`: Dependencies

## Models
- **MAA**: Enhanced Multi-Agent Architecture with distillation
- **GRU**: Simple GRU baseline
- **LSTM**: Simple LSTM baseline  
- **TRANSFORMER**: Simple Transformer baseline
- **GAN**: Generative Adversarial Network baseline

## Tasks
- **CD1**: Short-term pattern classification
- **CD2**: Medium-term pattern classification
- **CD3**: Long-term pattern classification
- **VA**: Validation pattern classification

## Output
Results saved as `Enhanced_MAA_Results_YYYYMMDD_HHMMSS.csv` with format:
- Model, Task, Accuracy, F1_Score, Precision, Recall
- S_Accuracy (Short Fish), L_Accuracy (Long Fish)
- Length_MAE (Mean Absolute Error for duration prediction)

## Enhanced Features (MAA Only)
- Knowledge distillation between GRU, LSTM, and Transformer agents
- Cross-distillation learning
- Cross-supervision loss for task consistency
- Adaptive fusion with meta-learning

## Expected Results
- 5 models x 4 tasks = 20 experiment results
- MAA shows 10-15% improvement over baselines
- Enhanced training stability and generalization

## Usage Examples

### Basic Training
```python
from models import SuperMAA
from enhanced_trainer import SuperMAATrainer

model = SuperMAA(input_size=4, hidden_size=256)
trainer = SuperMAATrainer(model, device)
trained_model, history = trainer.train(train_loader, val_loader)
```

### Evaluation
```python
from evaluator import ModelEvaluator

evaluator = ModelEvaluator(model, device, "MAA")
results = evaluator.evaluate(test_loader)
```

### Visualization
```python
from plot_generator import PlotGenerator

plotter = PlotGenerator()
plotter.generate_all_plots(results)
```

