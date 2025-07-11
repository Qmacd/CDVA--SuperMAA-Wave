import os
import sys
import argparse
from datetime import datetime

from prediction_output import PredictionOutputGenerator
from sequence_analysis import SequenceAnalyzer
from utils import setup_device, set_random_seed


def main():
    parser = argparse.ArgumentParser(description='Run prediction output and sequence analysis')
    parser.add_argument('--data_path', type=str, default='cvda_dataset', 
                       help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=3, 
                       help='Number of training epochs for models')
    parser.add_argument('--output_dir', type=str, default='prediction_outputs', 
                       help='Directory to save prediction outputs')
    parser.add_argument('--analysis_dir', type=str, default='sequence_analysis', 
                       help='Directory to save analysis results')
    parser.add_argument('--skip_training', action='store_true', 
                       help='Skip training and use existing models')
    parser.add_argument('--analysis_only', action='store_true', 
                       help='Run analysis only (requires existing prediction outputs)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Wave MAA Framework - Prediction Analysis Pipeline")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Arguments: {vars(args)}")
    
    set_random_seed(args.seed)
    
    device = setup_device()
    print(f"Using device: {device}")
    
    if not args.analysis_only and not os.path.exists(args.data_path):
        print(f"Error: Dataset path '{args.data_path}' not found!")
        print("Please ensure the cvda_dataset is placed in one of these locations:")
        print("  - ./cvda_dataset/")
        print("  - ./data/cvda_dataset/")
        print("  - /root/autodl-tmp/cvda_dataset/")
        print("  - /root/wave_maa/cvda_dataset/")
        print("  - /root/cvda_dataset/")
        return 1
    
    try:
        if not args.analysis_only:
            print(f"\n{'-' * 60}")
            print("Step 1: Generating Prediction Outputs")
            print(f"{'-' * 60}")
            
            generator = PredictionOutputGenerator(device, args.output_dir)
            
            if args.skip_training:
                print("Skipping training - using existing models")
                print("Warning: Skip training mode not fully implemented")
            
            predictions = generator.run_prediction_output(
                data_path=args.data_path, 
                epochs=args.epochs
            )
            
            print(f"Prediction outputs saved to: {args.output_dir}")
        
        print(f"\n{'-' * 60}")
        print("Step 2: Running Sequence Analysis")
        print(f"{'-' * 60}")
        
        analyzer = SequenceAnalyzer(args.analysis_dir)
        analysis_results = analyzer.run_full_analysis(args.output_dir)
        
        print(f"Analysis results saved to: {args.analysis_dir}")
        
        print(f"\n{'=' * 80}")
        print("Prediction Analysis Pipeline Completed!")
        print(f"{'=' * 80}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not args.analysis_only:
            print(f"Prediction outputs: {args.output_dir}")
        print(f"Analysis results: {args.analysis_dir}")
        
        print("\nGenerated files:")
        if os.path.exists(args.output_dir):
            pred_files = [f for f in os.listdir(args.output_dir) if f.endswith('.csv')]
            print(f"  Prediction files: {len(pred_files)}")
        
        if os.path.exists(args.analysis_dir):
            analysis_files = [f for f in os.listdir(args.analysis_dir) if f.endswith(('.csv', '.json'))]
            print(f"  Analysis files: {len(analysis_files)}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

