"""
Experiment Result Analysis for COMP3242/6242 Project
Reads metrics.csv, generates comparative plots of validation loss and training time.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================== CONFIGURATION ==================
DATA_FILE = "metrics.csv"          
OUTPUT_DIR = "figures"              
SUMMARY_CSV = "aggregated_metrics.csv"
FIG_DPI = 300                       
PLOT_STYLE = "whitegrid"            

# ================== DATA LOADING & CLEANING ==================
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    df = df[df['device'] == 'cpu'].copy()
    
    if 'use_pos_encoding' in df.columns:
        df.rename(columns={'use_pos_encoding': 'pos_encoding'}, inplace=True)
    elif 'pos_encoding' not in df.columns:
        pass
    
    valid_pos = ['learned', 'rope', 'alibi']
    valid_norm = ['layernorm', 'none']
    df = df[df['pos_encoding'].isin(valid_pos)]
    df = df[df['norm_type'].isin(valid_norm)]
    df = df[df['n_head'].isin([1, 6])]
    
    return df

def aggregate_results(df):
    group_cols = ['pos_encoding', 'norm_type', 'n_head']
    agg_dict = {
        'final_val_loss': ['mean', 'std', 'count'],
        'training_time_sec': ['mean', 'std']
    }
    aggregated = df.groupby(group_cols).agg(agg_dict).reset_index()
    aggregated.columns = ['pos_encoding', 'norm_type', 'n_head',
                          'val_loss_mean', 'val_loss_std', 'count',
                          'time_mean', 'time_std']
    aggregated['val_loss_std'] = aggregated['val_loss_std'].fillna(0)
    aggregated['time_std'] = aggregated['time_std'].fillna(0)
    return aggregated

# ================== PLOTTING FUNCTIONS ==================
def set_plot_style():
    """设置全局绘图样式"""
    sns.set_style(PLOT_STYLE)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = FIG_DPI

def plot_val_loss_bars(agg_df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    heads = [1, 6]
    
    for i, head in enumerate(heads):
        ax = axes[i]
        subset = agg_df[agg_df['n_head'] == head].copy()
        subset['config'] = subset['pos_encoding'] + "\n" + subset['norm_type']
        x_labels = subset['config']
        x_pos = np.arange(len(x_labels))
        means = subset['val_loss_mean']
        stds = subset['val_loss_std']
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color='skyblue', edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=0, ha='center')
        ax.set_ylabel("Validation Loss")
        ax.set_title(f"Number of Heads = {head}")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle("Final Validation Loss by Position Encoding and Normalization", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "val_loss_comparison.png"), dpi=FIG_DPI, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_dir}/val_loss_comparison.png")

def plot_training_time_bars(agg_df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    heads = [1, 6]
    
    for i, head in enumerate(heads):
        ax = axes[i]
        subset = agg_df[agg_df['n_head'] == head].copy()
        subset['config'] = subset['pos_encoding'] + "\n" + subset['norm_type']
        x_labels = subset['config']
        x_pos = np.arange(len(x_labels))
        means = subset['time_mean']
        stds = subset['time_std']
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color='lightcoral', edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Training Time (seconds)")
        ax.set_title(f"Number of Heads = {head}")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    f'{mean:.1f}s', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle("Training Time by Configuration", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_time_comparison.png"), dpi=FIG_DPI, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_dir}/training_time_comparison.png")

def plot_loss_vs_time_scatter(agg_df, output_dir):
    plt.figure(figsize=(10, 6))
    markers = {1: 'o', 6: 's'}
    colors = {1: 'blue', 6: 'red'}
    
    for head in [1, 6]:
        subset = agg_df[agg_df['n_head'] == head]
        plt.scatter(subset['time_mean'], subset['val_loss_mean'],
                    s=100, marker=markers[head], c=colors[head],
                    label=f'n_head = {head}', alpha=0.8, edgecolors='black')
        for _, row in subset.iterrows():
            label = f"{row['pos_encoding']}/{row['norm_type']}"
            plt.annotate(label, (row['time_mean'], row['val_loss_mean']),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel("Training Time (seconds)")
    plt.ylabel("Validation Loss")
    plt.title("Trade-off: Validation Loss vs Training Time")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_vs_time_scatter.png"), dpi=FIG_DPI)
    plt.show()
    print(f"Saved: {output_dir}/loss_vs_time_scatter.png")

def plot_heatmap_by_heads(agg_df, output_dir):
    heads = [1, 6]
    for head in heads:
        subset = agg_df[agg_df['n_head'] == head].pivot(index='norm_type',
                                                        columns='pos_encoding',
                                                        values='val_loss_mean')
        plt.figure(figsize=(6, 3))
        sns.heatmap(subset, annot=True, fmt='.3f', cmap='viridis_r',
                    cbar_kws={'label': 'Validation Loss'})
        plt.title(f"Validation Loss Heatmap (n_head = {head})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"heatmap_heads_{head}.png"), dpi=FIG_DPI)
        plt.show()
        print(f"Saved: {output_dir}/heatmap_heads_{head}.png")

def save_summary_table(agg_df, output_dir):
    agg_df.to_csv(os.path.join(output_dir, SUMMARY_CSV), index=False)
    print(f"Summary table saved to {output_dir}/{SUMMARY_CSV}")

# ================== MAIN PIPELINE ==================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading and cleaning data...")
    df_raw = load_and_clean_data(DATA_FILE)
    print(f"Loaded {len(df_raw)} valid experiment runs.")
    
    agg_df = aggregate_results(df_raw)
    print("Aggregated results:")
    print(agg_df.to_string())
    
    set_plot_style()
    plot_val_loss_bars(agg_df, OUTPUT_DIR)
    plot_training_time_bars(agg_df, OUTPUT_DIR)
    plot_loss_vs_time_scatter(agg_df, OUTPUT_DIR)
    plot_heatmap_by_heads(agg_df, OUTPUT_DIR)
    save_summary_table(agg_df, OUTPUT_DIR)
    
    print("\nAnalysis completed! All figures saved in '{}' directory.".format(OUTPUT_DIR))

if __name__ == "__main__":
    main()