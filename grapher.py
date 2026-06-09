import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class AblationAnalyzer:
    def __init__(self, csv_path):
        """Loads and cleans the Weights & Biases CSV export."""
        self.df = pd.read_csv(csv_path)
        
        # Ensure correct datatypes
        numeric_cols = ['Runtime', 'train/loss', 'val/loss', 'n_params']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
        # Create output directory for graphs
        os.makedirs("graphs", exist_ok=True)
        
        # Set Seaborn style for presentation-ready plots
        sns.set_theme(style="whitegrid", context="talk")
        
        # Aggregate data across the 4 seeds
        self.agg_df = self._aggregate_data()

    def _aggregate_data(self):
        """Groups by run_name to get mean and std dev across random seeds."""
        agg = self.df.groupby('run_name').agg(
            val_loss_mean=('val/loss', 'mean'),
            val_loss_std=('val/loss', 'std'),
            train_loss_mean=('train/loss', 'mean'),
            train_loss_std=('train/loss', 'std'),
            runtime_mean=('Runtime', 'mean'),
            runtime_std=('Runtime', 'std'),
            n_params=('n_params', 'first')
        ).reset_index()
        
        # Sort by validation loss by default
        return agg.sort_values('val_loss_mean')

    def plot_val_loss_ranking(self, save_name="val_loss_ranking.png"):
        """Plots a ranked bar chart of all models by Validation Loss."""
        plt.figure(figsize=(14, 8))
        
        # Drop the "failed" no_residual model to not skew the Y-axis scale
        plot_df = self.agg_df[self.agg_df['run_name'] != 'E2_no_residuals']
        
        # Color baseline gray, ternary green, others blue
        colors = ['#2ecc71' if 'ternary' in name else '#95a5a6' if 'baseline' in name else '#3498db' 
                  for name in plot_df['run_name']]

        bars = plt.bar(plot_df['run_name'], plot_df['val_loss_mean'], 
                       yerr=plot_df['val_loss_std'], capsize=5, color=colors, alpha=0.8)
        
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.ylim(1.4, 2.1) # Zoom in to show differences
        plt.ylabel('Validation Loss (Lower is Better)')
        plt.title('Overall Model Performance (Ranked by Validation Loss)')
        plt.tight_layout()
        plt.savefig(f"graphs/{save_name}", dpi=300)
        plt.close()
        print(f"Saved {save_name}")

    def plot_overfitting_gap(self, runs_to_include, title, save_name="overfitting.png"):
        """Plots Train vs Val loss side-by-side to visualize overfitting."""
        filtered_df = self.agg_df[self.agg_df['run_name'].isin(runs_to_include)].copy()
        
        # Define x-axis positions
        x = np.arange(len(filtered_df['run_name']))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, filtered_df['train_loss_mean'], width, 
                yerr=filtered_df['train_loss_std'], label='Train Loss', color='#2c3e50', capsize=4)
        plt.bar(x + width/2, filtered_df['val_loss_mean'], width, 
                yerr=filtered_df['val_loss_std'], label='Validation Loss', color='#e74c3c', capsize=4)
        
        plt.xticks(x, filtered_df['run_name'], rotation=15, fontsize=11)
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"graphs/{save_name}", dpi=300)
        plt.close()
        print(f"Saved {save_name}")

    def plot_speed_vs_quality(self, save_name="speed_vs_quality.png"):
        """Scatter plot of Runtime vs Validation Loss."""
        plt.figure(figsize=(10, 7))
        
        # Remove the outlier failed model
        plot_df = self.agg_df[self.agg_df['run_name'] != 'E2_no_residuals']
        
        sns.scatterplot(data=plot_df, x='runtime_mean', y='val_loss_mean', 
                        s=150, color='#3498db', edgecolor='black')
        
        # Annotate points
        for i, row in plot_df.iterrows():
            # Only label important ones to avoid clutter
            if 'ternary' in row['run_name'] or 'baseline' in row['run_name'] or '128' in row['run_name'] or '512' in row['run_name'] or 'alibi' in row['run_name']:
                plt.text(row['runtime_mean'] + 15, row['val_loss_mean'], 
                         row['run_name'].replace('_', ' '), fontsize=9)
                
        plt.xlabel('Runtime (Seconds) -> Faster is Better')
        plt.ylabel('Validation Loss -> Lower is Better')
        plt.title('Speed vs. Quality Trade-off')
        
        # Draw a quadrant line for Baseline
        base_runtime = plot_df[plot_df['run_name'] == 'baseline']['runtime_mean'].values[0]
        base_val = plot_df[plot_df['run_name'] == 'baseline']['val_loss_mean'].values[0]
        plt.axvline(base_runtime, color='gray', linestyle='--', alpha=0.5)
        plt.axhline(base_val, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"graphs/{save_name}", dpi=300)
        plt.close()
        print(f"Saved {save_name}")

    def print_summary_statistics(self):
        """Prints a text report of the best and worst findings."""
        print("\n--- ABLATION STUDY SUMMARY ---")
        best_model = self.agg_df.iloc[0]
        print(f"🏆 Best Model: {best_model['run_name']} (Val Loss: {best_model['val_loss_mean']:.4f})")
        
        fastest_model = self.agg_df.sort_values('runtime_mean').iloc[0]
        print(f"⚡ Fastest Model: {fastest_model['run_name']} (Time: {fastest_model['runtime_mean']:.1f}s)")
        
        print("\nOverfitting Rankings (Gap between Train and Val):")
        self.agg_df['overfit_gap'] = self.agg_df['val_loss_mean'] - self.agg_df['train_loss_mean']
        gap_df = self.agg_df.sort_values('overfit_gap', ascending=False)
        for _, row in gap_df.head(5).iterrows():
            print(f"  - {row['run_name']}: Gap of {row['overfit_gap']:.4f}")
        print("------------------------------\n")


# ==========================================
# EXECUTION SCRIPT (Generate Report Graphs)
# ==========================================
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = AblationAnalyzer("wandb_export.csv") # Replace with your CSV filename
    
    # 1. Print textual summary for your script notes
    analyzer.print_summary_statistics()
    
    # 2. Graph 1: Overall Ranking (Highlights Ternary as the winner)
    analyzer.plot_val_loss_ranking("1_overall_ranking.png")
    
    # 3. Graph 2: The "Overfitting Trap" (High Capacity vs Baseline)
    # This directly supports Slide 6 of your new script!
    trap_runs = ['baseline', 'D2_swiglu', 'F2_context_512', 'C3_12_heads']
    analyzer.plot_overfitting_gap(
        runs_to_include=trap_runs, 
        title="The Overfitting Trap: High Capacity Harms Generalization",
        save_name="2_overfitting_trap.png"
    )
    
    # 4. Graph 3: Architecture as Regularization (Low Capacity / Ternary vs Baseline)
    # This supports Slide 8 of your script
    reg_runs = ['baseline', 'G1_ternary_weights', 'C1_1_head', 'F1_context_128']
    analyzer.plot_overfitting_gap(
        runs_to_include=reg_runs, 
        title="Architecture as Regularizer: Strict Constraints Improve Generalization",
        save_name="3_regularization_heroes.png"
    )
    
    # 5. Graph 4: Speed vs Quality (The Pareto Front)
    # This supports Slide 9
    analyzer.plot_speed_vs_quality("4_speed_vs_quality.png")