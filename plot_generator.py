import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('default')
            print("Using default matplotlib style (seaborn styles not available)")

try:
    sns.set_palette("husl")
except:
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

class WaveMAA_PlotGenerator:
    """Generate comprehensive plots for Wave MAA experiment results"""
    
    def __init__(self, results_csv_path, output_dir="plots"):
        """
        Initialize plot generator
        
        Args:
            results_csv_path: Path to experiment results CSV file
            output_dir: Directory to save plots
        """
        self.results_df = pd.read_csv(results_csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("Generating comprehensive visualization plots...")
        
        self.plot_model_performance()
        
        self.plot_task_performance()
        
        self.plot_accuracy_distribution()
        
        self.plot_performance_heatmap()
        
        self.plot_model_ranking()
        
        self.plot_fish_classification()
        
        self.plot_length_prediction()
        
        self.plot_comprehensive_dashboard()
        
        print(f"All plots saved to: {self.output_dir}")
        
    def plot_model_performance(self):
        """Plot model performance comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        model_acc = self.results_df.groupby('Model')['Accuracy'].mean().sort_values(ascending=False)
        axes[0,0].bar(model_acc.index, model_acc.values, color='skyblue', alpha=0.8)
        axes[0,0].set_title('Average Accuracy by Model')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        model_f1 = self.results_df.groupby('Model')['F1_Score'].mean().sort_values(ascending=False)
        axes[0,1].bar(model_f1.index, model_f1.values, color='lightcoral', alpha=0.8)
        axes[0,1].set_title('Average F1 Score by Model')
        axes[0,1].set_ylabel('F1 Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        for model in self.results_df['Model'].unique():
            model_data = self.results_df[self.results_df['Model'] == model]
            axes[1,0].scatter(model_data['Precision'], model_data['Recall'], 
                            label=model, alpha=0.7, s=60)
        axes[1,0].set_xlabel('Precision')
        axes[1,0].set_ylabel('Recall')
        axes[1,0].set_title('Precision vs Recall')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        model_mae = self.results_df.groupby('Model')['Length_MAE'].mean().sort_values()
        axes[1,1].bar(model_mae.index, model_mae.values, color='lightgreen', alpha=0.8)
        axes[1,1].set_title('Average Length MAE by Model (Lower is Better)')
        axes[1,1].set_ylabel('Length MAE')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'model_performance_{self.timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_task_performance(self):
        """Plot task-wise performance analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Task-wise Performance Analysis', fontsize=16, fontweight='bold')
        
        task_acc = self.results_df.groupby('Task')['Accuracy'].mean().sort_values(ascending=False)
        axes[0,0].bar(task_acc.index, task_acc.values, color='gold', alpha=0.8)
        axes[0,0].set_title('Average Accuracy by Task')
        axes[0,0].set_ylabel('Accuracy')
        
        self.results_df.boxplot(column='Accuracy', by='Task', ax=axes[0,1])
        axes[0,1].set_title('Accuracy Distribution by Task')
        axes[0,1].set_xlabel('Task')
        axes[0,1].set_ylabel('Accuracy')
        
        fish_data = []
        for _, row in self.results_df.iterrows():
            fish_data.append({'Task': row['Task'], 'Type': 'Short Fish', 'Accuracy': row['S_Accuracy']})
            fish_data.append({'Task': row['Task'], 'Type': 'Long Fish', 'Accuracy': row['L_Accuracy']})
        
        fish_df = pd.DataFrame(fish_data)
        sns.barplot(data=fish_df, x='Task', y='Accuracy', hue='Type', ax=axes[1,0])
        axes[1,0].set_title('Fish Classification Accuracy by Task')
        axes[1,0].legend(title='Fish Type')
        
        task_difficulty = self.results_df.groupby('Task').agg({
            'Accuracy': 'mean',
            'F1_Score': 'mean',
            'Length_MAE': 'mean'
        }).round(4)
        
        task_difficulty['Difficulty_Score'] = (1 - task_difficulty['Accuracy']) + (task_difficulty['Length_MAE'] / 10)
        task_difficulty = task_difficulty.sort_values('Difficulty_Score', ascending=False)
        
        axes[1,1].barh(task_difficulty.index, task_difficulty['Difficulty_Score'], color='salmon', alpha=0.8)
        axes[1,1].set_title('Task Difficulty Ranking')
        axes[1,1].set_xlabel('Difficulty Score (Higher = More Difficult)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'task_performance_{self.timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_accuracy_distribution(self):
        """Plot accuracy distribution analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Accuracy Distribution Analysis', fontsize=16, fontweight='bold')
        
        axes[0,0].hist(self.results_df['Accuracy'], bins=15, color='lightblue', alpha=0.7, edgecolor='black')
        axes[0,0].set_title('Overall Accuracy Distribution')
        axes[0,0].set_xlabel('Accuracy')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(self.results_df['Accuracy'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {self.results_df["Accuracy"].mean():.3f}')
        axes[0,0].legend()
        
        sns.violinplot(data=self.results_df, x='Model', y='Accuracy', ax=axes[0,1])
        axes[0,1].set_title('Accuracy Distribution by Model')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        sorted_acc = np.sort(self.results_df['Accuracy'])
        cumulative = np.arange(1, len(sorted_acc) + 1) / len(sorted_acc)
        axes[1,0].plot(sorted_acc, cumulative, marker='o', linestyle='-', alpha=0.7)
        axes[1,0].set_title('Cumulative Accuracy Distribution')
        axes[1,0].set_xlabel('Accuracy')
        axes[1,0].set_ylabel('Cumulative Probability')
        axes[1,0].grid(True, alpha=0.3)
        
        corr_cols = ['Accuracy', 'F1_Score', 'Precision', 'Recall', 'S_Accuracy', 'L_Accuracy', 'Length_MAE']
        corr_matrix = self.results_df[corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
        axes[1,1].set_title('Performance Metrics Correlation')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'accuracy_distribution_{self.timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_performance_heatmap(self):
        """Plot performance heatmap"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Performance Heatmap Analysis', fontsize=16, fontweight='bold')
        
        accuracy_pivot = self.results_df.pivot(index='Model', columns='Task', values='Accuracy')
        sns.heatmap(accuracy_pivot, annot=True, cmap='RdYlGn', center=0.7, 
                   fmt='.3f', ax=axes[0])
        axes[0].set_title('Accuracy Heatmap (Model vs Task)')
        
        f1_pivot = self.results_df.pivot(index='Model', columns='Task', values='F1_Score')
        sns.heatmap(f1_pivot, annot=True, cmap='RdYlGn', center=0.7, 
                   fmt='.3f', ax=axes[1])
        axes[1].set_title('F1 Score Heatmap (Model vs Task)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'performance_heatmap_{self.timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_model_ranking(self):
        """Plot model ranking analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Ranking Analysis', fontsize=16, fontweight='bold')
        
        model_ranking = self.results_df.groupby('Model')['Accuracy'].mean().sort_values(ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(model_ranking)))
        axes[0,0].barh(model_ranking.index, model_ranking.values, color=colors)
        axes[0,0].set_title('Model Ranking by Average Accuracy')
        axes[0,0].set_xlabel('Average Accuracy')
        
        metrics = ['Accuracy', 'F1_Score', 'Precision', 'Recall']
        model_scores = self.results_df.groupby('Model')[metrics].mean()
        
        normalized_scores = (model_scores - model_scores.min()) / (model_scores.max() - model_scores.min())
        normalized_scores['Overall_Score'] = normalized_scores.mean(axis=1)
        normalized_scores = normalized_scores.sort_values('Overall_Score', ascending=True)
        
        axes[0,1].barh(normalized_scores.index, normalized_scores['Overall_Score'], 
                      color='lightcoral', alpha=0.8)
        axes[0,1].set_title('Model Ranking by Overall Score')
        axes[0,1].set_xlabel('Normalized Overall Score')
        
        top_3_models = model_scores.mean(axis=1).nlargest(3).index
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax_radar = plt.subplot(2, 2, 3, projection='polar')
        
        for i, model in enumerate(top_3_models):
            values = model_scores.loc[model, metrics].tolist()
            values += values[:1]  # Complete the circle
            ax_radar.plot(angles, values, 'o-', linewidth=2, label=model)
            ax_radar.fill(angles, values, alpha=0.25)
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics)
        ax_radar.set_title('Top 3 Models Performance Radar')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        win_counts = []
        for task in self.results_df['Task'].unique():
            task_data = self.results_df[self.results_df['Task'] == task]
            best_model = task_data.loc[task_data['Accuracy'].idxmax(), 'Model']
            win_counts.append({'Task': task, 'Best_Model': best_model})
        
        win_df = pd.DataFrame(win_counts)
        win_rate = win_df['Best_Model'].value_counts()
        
        axes[1,1].pie(win_rate.values, labels=win_rate.index, autopct='%1.1f%%', startangle=90)
        axes[1,1].set_title('Task Win Rate by Model')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'model_ranking_{self.timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_fish_classification(self):
        """Plot fish classification analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fish Classification Analysis', fontsize=16, fontweight='bold')
        
        fish_comparison = []
        for _, row in self.results_df.iterrows():
            fish_comparison.append({
                'Model': row['Model'], 
                'Task': row['Task'],
                'Short_Fish_Acc': row['S_Accuracy'],
                'Long_Fish_Acc': row['L_Accuracy'],
                'Difference': row['L_Accuracy'] - row['S_Accuracy']
            })
        
        fish_df = pd.DataFrame(fish_comparison)
        
        avg_short = fish_df['Short_Fish_Acc'].mean()
        avg_long = fish_df['Long_Fish_Acc'].mean()
        
        axes[0,0].bar(['Short Fish', 'Long Fish'], [avg_short, avg_long], 
                     color=['lightblue', 'lightcoral'], alpha=0.8)
        axes[0,0].set_title('Average Classification Accuracy by Fish Type')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_ylim(0, 1)
        
        model_diff = fish_df.groupby('Model')['Difference'].mean().sort_values()
        colors = ['red' if x < 0 else 'green' for x in model_diff.values]
        axes[0,1].barh(model_diff.index, model_diff.values, color=colors, alpha=0.7)
        axes[0,1].set_title('Long Fish vs Short Fish Accuracy Difference')
        axes[0,1].set_xlabel('Accuracy Difference (Long - Short)')
        axes[0,1].axvline(0, color='black', linestyle='--', alpha=0.5)
        
        task_fish_data = []
        for task in self.results_df['Task'].unique():
            task_data = self.results_df[self.results_df['Task'] == task]
            task_fish_data.append({
                'Task': task,
                'Short_Fish': task_data['S_Accuracy'].mean(),
                'Long_Fish': task_data['L_Accuracy'].mean()
            })
        
        task_fish_df = pd.DataFrame(task_fish_data)
        x = np.arange(len(task_fish_df))
        width = 0.35
        
        axes[1,0].bar(x - width/2, task_fish_df['Short_Fish'], width, 
                     label='Short Fish', color='lightblue', alpha=0.8)
        axes[1,0].bar(x + width/2, task_fish_df['Long_Fish'], width, 
                     label='Long Fish', color='lightcoral', alpha=0.8)
        axes[1,0].set_title('Fish Classification Accuracy by Task')
        axes[1,0].set_xlabel('Task')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(task_fish_df['Task'])
        axes[1,0].legend()
        
        axes[1,1].scatter(fish_df['Short_Fish_Acc'], fish_df['Long_Fish_Acc'], 
                         alpha=0.6, s=60)
        axes[1,1].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Equal Performance')
        axes[1,1].set_xlabel('Short Fish Accuracy')
        axes[1,1].set_ylabel('Long Fish Accuracy')
        axes[1,1].set_title('Short Fish vs Long Fish Accuracy')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'fish_classification_{self.timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_length_prediction(self):
        """Plot length prediction analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fish Length Prediction Analysis', fontsize=16, fontweight='bold')
        
        model_mae = self.results_df.groupby('Model')['Length_MAE'].mean().sort_values()
        axes[0,0].bar(model_mae.index, model_mae.values, color='lightgreen', alpha=0.8)
        axes[0,0].set_title('Average Length MAE by Model')
        axes[0,0].set_ylabel('Length MAE (days)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        task_mae = self.results_df.groupby('Task')['Length_MAE'].mean().sort_values()
        axes[0,1].bar(task_mae.index, task_mae.values, color='gold', alpha=0.8)
        axes[0,1].set_title('Average Length MAE by Task')
        axes[0,1].set_ylabel('Length MAE (days)')
        
        axes[1,0].hist(self.results_df['Length_MAE'], bins=15, color='salmon', alpha=0.7, edgecolor='black')
        axes[1,0].set_title('Length MAE Distribution')
        axes[1,0].set_xlabel('Length MAE (days)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].axvline(self.results_df['Length_MAE'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {self.results_df["Length_MAE"].mean():.2f}')
        axes[1,0].legend()
        
        axes[1,1].scatter(self.results_df['Accuracy'], self.results_df['Length_MAE'], 
                         alpha=0.6, s=60)
        axes[1,1].set_xlabel('Accuracy')
        axes[1,1].set_ylabel('Length MAE (days)')
        axes[1,1].set_title('Accuracy vs Length MAE Correlation')
        
        corr = self.results_df['Accuracy'].corr(self.results_df['Length_MAE'])
        axes[1,1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                      transform=axes[1,1].transAxes, fontsize=12,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'length_prediction_{self.timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_comprehensive_dashboard(self):
        """Plot comprehensive dashboard"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        fig.suptitle('Wave MAA Experiment - Comprehensive Dashboard', fontsize=20, fontweight='bold')
        
        ax1 = fig.add_subplot(gs[0, :2])
        model_acc = self.results_df.groupby('Model')['Accuracy'].mean().sort_values(ascending=False)
        bars = ax1.bar(model_acc.index, model_acc.values, color='skyblue', alpha=0.8)
        ax1.set_title('Model Performance Overview', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        ax2 = fig.add_subplot(gs[0, 2:])
        task_acc = self.results_df.groupby('Task')['Accuracy'].mean().sort_values(ascending=False)
        ax2.bar(task_acc.index, task_acc.values, color='lightcoral', alpha=0.8)
        ax2.set_title('Task Difficulty Analysis', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Accuracy')
        
        ax3 = fig.add_subplot(gs[1, :2])
        accuracy_pivot = self.results_df.pivot(index='Model', columns='Task', values='Accuracy')
        sns.heatmap(accuracy_pivot, annot=True, cmap='RdYlGn', center=0.7, 
                   fmt='.3f', ax=ax3, cbar_kws={'shrink': 0.8})
        ax3.set_title('Accuracy Heatmap', fontsize=14, fontweight='bold')
        
        ax4 = fig.add_subplot(gs[1, 2:])
        fish_data = []
        for _, row in self.results_df.iterrows():
            fish_data.append({'Type': 'Short Fish', 'Accuracy': row['S_Accuracy']})
            fish_data.append({'Type': 'Long Fish', 'Accuracy': row['L_Accuracy']})
        fish_df = pd.DataFrame(fish_data)
        sns.boxplot(data=fish_df, x='Type', y='Accuracy', ax=ax4)
        ax4.set_title('Fish Classification Performance', fontsize=14, fontweight='bold')
        
        ax5 = fig.add_subplot(gs[2, :2])
        model_ranking = self.results_df.groupby('Model')['Accuracy'].mean().sort_values(ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(model_ranking)))
        ax5.barh(model_ranking.index, model_ranking.values, color=colors)
        ax5.set_title('Model Ranking', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Average Accuracy')
        
        ax6 = fig.add_subplot(gs[2, 2:])
        model_mae = self.results_df.groupby('Model')['Length_MAE'].mean().sort_values()
        ax6.bar(model_mae.index, model_mae.values, color='lightgreen', alpha=0.8)
        ax6.set_title('Length Prediction Performance', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Length MAE (days)')
        ax6.tick_params(axis='x', rotation=45)
        
        ax7 = fig.add_subplot(gs[3, :2])
        ax7.hist(self.results_df['Accuracy'], bins=15, color='lightblue', alpha=0.7, edgecolor='black')
        ax7.set_title('Accuracy Distribution', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Accuracy')
        ax7.set_ylabel('Frequency')
        ax7.axvline(self.results_df['Accuracy'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.results_df["Accuracy"].mean():.3f}')
        ax7.legend()
        
        ax8 = fig.add_subplot(gs[3, 2:])
        ax8.axis('off')
        
        best_model = self.results_df.groupby('Model')['Accuracy'].mean().idxmax()
        best_acc = self.results_df.groupby('Model')['Accuracy'].mean().max()
        worst_model = self.results_df.groupby('Model')['Accuracy'].mean().idxmin()
        worst_acc = self.results_df.groupby('Model')['Accuracy'].mean().min()
        
        maa_acc = self.results_df[self.results_df['Model'] == 'MAA']['Accuracy'].mean() if 'MAA' in self.results_df['Model'].values else 0
        other_acc = self.results_df[self.results_df['Model'] != 'MAA']['Accuracy'].mean()
        
        summary_text = f"""
        EXPERIMENT SUMMARY
        
        Best Model: {best_model} ({best_acc:.3f})
        Worst Model: {worst_model} ({worst_acc:.3f})
        
        MAA Performance: {maa_acc:.3f}
        Other Models Avg: {other_acc:.3f}
        MAA Advantage: {((maa_acc-other_acc)/other_acc*100):.1f}%
        
        Total Experiments: {len(self.results_df)}
        Models Tested: {self.results_df['Model'].nunique()}
        Tasks Evaluated: {self.results_df['Task'].nunique()}
        
        Overall Statistics:
        • Mean Accuracy: {self.results_df['Accuracy'].mean():.3f}
        • Std Accuracy: {self.results_df['Accuracy'].std():.3f}
        • Mean F1 Score: {self.results_df['F1_Score'].mean():.3f}
        • Mean Length MAE: {self.results_df['Length_MAE'].mean():.2f} days
        """
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
        
        plt.savefig(self.output_dir / f'comprehensive_dashboard_{self.timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive dashboard saved")

def generate_plots_from_csv(csv_path, output_dir="plots"):
    """
    Generate all plots from experiment results CSV
    
    Args:
        csv_path: Path to experiment results CSV file
        output_dir: Directory to save plots
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    
    generator = WaveMAA_PlotGenerator(csv_path, output_dir)
    generator.generate_all_plots()
    
    return generator.output_dir

if __name__ == "__main__":
    csv_files = [f for f in os.listdir('.') if f.startswith('AutoDL_MAA_Results_') and f.endswith('.csv')]
    
    if csv_files:
        latest_csv = sorted(csv_files)[-1]
        print(f"Found results file: {latest_csv}")
        
        output_dir = generate_plots_from_csv(latest_csv)
        print(f"All plots generated successfully!")
        print(f"Plots saved to: {output_dir}")
    else:
        print("No AutoDL_MAA_Results_*.csv file found in current directory")
        print("Please run the experiment first to generate results")
