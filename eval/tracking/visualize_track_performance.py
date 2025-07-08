import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_clean_data(csv_path):
    """Load and clean the track performance data."""
    df = pd.read_csv(csv_path)
    df['event_idx'] = df['event_idx'].astype(str).str.replace('tensor(', '').str.replace(')', '').astype(int)
    
    print(f"Loaded {len(df)} tracks from {len(df['event_idx'].unique())} events")
    print(f"Track length range: {df['track_length'].min()} - {df['track_length'].max()}")
    print(f"Average pt range: {df['avg_pt_in_track'].min():.2f} - {df['avg_pt_in_track'].max():.2f}")
    
    return df

def create_performance_overview(df, save_dir):
    """Create overview plots of track performance metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Track Performance Overview', fontsize=16, fontweight='bold')
    
    df_copy = df.copy()
    df_copy['track_bin'] = pd.cut(df_copy['track_length'], bins=[2, 6, 10, float('inf')], 
                                  labels=['2-6', '6-10', '10+'], right=False)
    
    def plot_metric_by_track_length(ax, metric_name, title, bins):
        """Plot metric with mean and median lines for each track length bin."""
        ax.hist(df[metric_name], bins=bins, alpha=0.7, edgecolor='black')
        
        colors = ['red', 'blue', 'green']
        for i, (bin_name, color) in enumerate(zip(['2-6', '6-10', '10+'], colors)):
            bin_data = df_copy[df_copy['track_bin'] == bin_name][metric_name]
            if len(bin_data) > 0:
                mean_val = bin_data.mean()
                median_val = bin_data.median()
                ax.axvline(mean_val, color=color, linestyle='--', alpha=0.8,
                          label=f'{bin_name} Mean: {mean_val:.3f}')
                ax.axvline(median_val, color=color, linestyle=':', alpha=0.8,
                          label=f'{bin_name} Median: {median_val:.3f}')
        
        ax.set_xlabel(metric_name.capitalize())
        ax.set_ylabel('Number of Tracks')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    precision_bins = np.linspace(df['precision'].min(), df['precision'].max(), 21)
    plot_metric_by_track_length(axes[0, 0], 'precision', 'Track Precision Distribution (20 bins)', precision_bins)
    
    recall_bins = np.linspace(df['recall'].min(), df['recall'].max(), 21)
    plot_metric_by_track_length(axes[0, 1], 'recall', 'Track Recall Distribution (20 bins)', recall_bins)
    
    completeness_bins = np.linspace(0, 1, 101)
    plot_metric_by_track_length(axes[0, 2], 'completeness', 'Track Completeness Distribution (0-1, 100 bins)', completeness_bins)
    axes[0, 2].set_xlim(0, 1)
    
    purity_bins = np.linspace(0, 1, 101)
    plot_metric_by_track_length(axes[1, 0], 'purity', 'Track Purity Distribution (0-1, 100 bins)', purity_bins)
    axes[1, 0].set_xlim(0, 1)
    
    track_length_bins = np.linspace(1, 40, 41)
    axes[1, 1].hist(df['track_length'], bins=track_length_bins, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(df['track_length'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["track_length"].mean():.1f}')
    axes[1, 1].axvline(df['track_length'].median(), color='blue', linestyle=':', 
                       label=f'Median: {df["track_length"].median():.1f}')
    axes[1, 1].set_xlabel('Track Length')
    axes[1, 1].set_ylabel('Number of Tracks')
    axes[1, 1].set_title('Track Length Distribution (1-40)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(1, 40)
    
    pt_bins = np.linspace(0, 10, 101)
    plot_metric_by_track_length(axes[1, 2], 'avg_pt_in_track', 'Average pT Distribution (0-10)', pt_bins)
    axes[1, 2].set_xlim(0, 10)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'performance_overview.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_performance_by_track_length(df, save_dir):
    """Analyze performance as a function of track length."""
    track_length_bins = [2, 3, 4, 5, 7, 10, 15, 25, max(25.1, df['track_length'].max()+1)]
    df['length_bin'] = pd.cut(df['track_length'], bins=track_length_bins, right=False)
    
    stats_by_length = df.groupby('length_bin', observed=False).agg({
        'precision': ['mean', 'std', 'count'],
        'recall': ['mean', 'std', 'count'],
        'completeness': ['mean', 'std', 'count'],
        'purity': ['mean', 'std', 'count']
    }).round(3)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance vs Track Length', fontsize=16, fontweight='bold')
    
    metrics = ['precision', 'recall', 'completeness', 'purity']
    titles = ['Precision vs Track Length', 'Recall vs Track Length', 
              'Completeness vs Track Length', 'Purity vs Track Length']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i//2, i%2]
        
        box_data = [group[metric].values for name, group in df.groupby('length_bin', observed=False)]
        box_labels = [str(name) for name, group in df.groupby('length_bin', observed=False)]
        
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', color='blue', linewidth=2),
                       whiskerprops=dict(color='blue', linewidth=2),
                       capprops=dict(color='blue', linewidth=2),
                       medianprops=dict(color='red', linewidth=2))
        
        ax.set_title(title)
        ax.set_xlabel('Track Length Range')
        ax.set_ylabel(metric.capitalize())
        ax.grid(True, alpha=0.3)
        
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'performance_by_track_length.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return stats_by_length

def create_performance_by_pt(df, save_dir):
    """Analyze performance as a function of average pT."""

    pt_bins = [0, 1, 2, 5, 10, 20, 50, max(50.1, df['avg_pt_in_track'].max())]
    df['pt_bin'] = pd.cut(df['avg_pt_in_track'], bins=pt_bins, right=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance vs Average pT', fontsize=16, fontweight='bold')
    
    metrics = ['precision', 'recall', 'completeness', 'purity']
    titles = ['Precision vs Average pT', 'Recall vs Average pT', 
              'Completeness vs Average pT', 'Purity vs Average pT']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i//2, i%2]
        
        box_data = [group[metric].values for name, group in df.groupby('pt_bin', observed=False)]
        box_labels = [str(name) for name, group in df.groupby('pt_bin', observed=False)]
        
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, 
                       boxprops=dict(facecolor='lightcoral', color='red', linewidth=2),
                       whiskerprops=dict(color='red', linewidth=2),
                       capprops=dict(color='red', linewidth=2),
                       medianprops=dict(color='darkblue', linewidth=2))
        
        ax.set_title(title)
        ax.set_xlabel('Average pT Range')
        ax.set_ylabel(metric.capitalize())
        ax.grid(True, alpha=0.3)
        
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'performance_by_pt.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_confusion_analysis(df, save_dir):
    """Analyze the confusion matrix elements (TP, FP, FN)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Confusion Matrix Analysis', fontsize=16, fontweight='bold')
    
    confusion_metrics = ['average_tp', 'average_fp', 'average_fn']
    confusion_titles = ['Average True Positives per Point', 'Average False Positives per Point', 
                       'Average False Negatives per Point']
    
    for i, (metric, title) in enumerate(zip(confusion_metrics, confusion_titles)):
        ax = axes[0, i]
        avg_bins = np.linspace(1, 40, 41)
        ax.hist(df[metric], bins=avg_bins, alpha=0.7, edgecolor='black')
        ax.axvline(df[metric].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df[metric].mean():.2f}')
        ax.set_xlabel(title.split(' per Point')[0])
        ax.set_ylabel('Number of Tracks')
        ax.set_title(title)
        ax.set_xlim(1, 40)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    total_metrics = ['total_tp', 'total_fp', 'total_fn']
    total_titles = ['Total True Positives', 'Total False Positives', 'Total False Negatives']
    
    for i, (metric, title) in enumerate(zip(total_metrics, total_titles)):
        ax = axes[1, i]
        min_val = df[metric].quantile(0.0)
        max_val = df[metric].quantile(0.95)
        total_bins = np.linspace(min_val, max_val, 41)
        
        ax.hist(df[metric], bins=total_bins, alpha=0.7, edgecolor='black')
        ax.axvline(df[metric].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df[metric].mean():.1f}')
        ax.set_xlabel(title)
        ax.set_ylabel('Number of Tracks')
        ax.set_title(f'{title} (0-95th percentile)')
        ax.set_xlim(min_val, max_val)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

def identify_problem_tracks(df, save_dir):
    """Identify and analyze problematic tracks."""

    low_precision = df['precision'] < 0.5
    low_recall = df['recall'] < 0.5
    low_completeness = df['completeness'] < 0.5
    
    problem_tracks = df[low_precision | low_recall | low_completeness].copy()
    
    print(f"\nProblem Track Analysis:")
    print("="*50)
    print(f"Total tracks: {len(df)}")
    print(f"Problem tracks: {len(problem_tracks)} ({len(problem_tracks)/len(df)*100:.1f}%)")
    print(f"Low precision (<0.5): {low_precision.sum()} ({low_precision.sum()/len(df)*100:.1f}%)")
    print(f"Low recall (<0.5): {low_recall.sum()} ({low_recall.sum()/len(df)*100:.1f}%)")
    print(f"Low completeness (<0.5): {low_completeness.sum()} ({low_completeness.sum()/len(df)*100:.1f}%)")
    
    if len(problem_tracks) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Problem Track Analysis', fontsize=16, fontweight='bold')
        
        track_length_bins = np.linspace(1, 40, 41)
        axes[0, 0].hist(df['track_length'], bins=track_length_bins, alpha=0.5, 
                        label='All tracks', density=True)
        axes[0, 0].hist(problem_tracks['track_length'], bins=track_length_bins, alpha=0.7, 
                        label='Problem tracks', density=True)
        axes[0, 0].set_xlabel('Track Length')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Track Length Distribution (1-40)')
        axes[0, 0].set_xlim(1, 40)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        pt_bins = np.linspace(0, 10, 101)
        axes[0, 1].hist(df['avg_pt_in_track'], bins=pt_bins, alpha=0.5, 
                        label='All tracks', density=True)
        axes[0, 1].hist(problem_tracks['avg_pt_in_track'], bins=pt_bins, alpha=0.7, 
                        label='Problem tracks', density=True)
        axes[0, 1].set_xlabel('Average pT')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Average pT Distribution (0-10)')
        axes[0, 1].set_xlim(0, 10)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        worst_precision = problem_tracks.nsmallest(10, 'precision')
        axes[1, 0].barh(range(len(worst_precision)), worst_precision['precision'])
        axes[1, 0].set_yticks(range(len(worst_precision)))
        axes[1, 0].set_yticklabels([f"Track {row['unique_track_id']}" for _, row in worst_precision.iterrows()])
        axes[1, 0].set_xlabel('Precision')
        axes[1, 0].set_title('10 Worst Tracks by Precision')
        axes[1, 0].grid(True, alpha=0.3)
        
        worst_recall = problem_tracks.nsmallest(10, 'recall')
        axes[1, 1].barh(range(len(worst_recall)), worst_recall['recall'])
        axes[1, 1].set_yticks(range(len(worst_recall)))
        axes[1, 1].set_yticklabels([f"Track {row['unique_track_id']}" for _, row in worst_recall.iterrows()])
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_title('10 Worst Tracks by Recall')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'problem_tracks_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)  
        
        problem_tracks.to_csv(save_dir / 'problem_tracks.csv', index=False)
        print(f"Problem tracks saved to: {save_dir / 'problem_tracks.csv'}")

def create_summary_statistics(df, save_dir):
    """Create a comprehensive summary of statistics including noise metrics."""
    summary_stats = df[['precision', 'recall', 'completeness', 'purity', 
                       'track_length', 'avg_pt_in_track', 'total_tp', 'total_fp', 'total_fn',
                       'noise_contamination_rate', 'track_cluster_purity', 'avg_noise_in_knn',
                       'total_noise_in_knn', 'total_other_tracks_in_knn']].describe()
    
    print("\nSummary Statistics (including noise metrics):")
    print("="*100)
    print(summary_stats.round(3))
    
    print(f"\nNOISE CONTAMINATION ANALYSIS:")
    print(f"="*50)
    print(f"Tracks with low noise contamination (≤0.2): {(df['noise_contamination_rate'] <= 0.2).sum()} ({(df['noise_contamination_rate'] <= 0.2).mean()*100:.1f}%)")
    print(f"Tracks with medium noise contamination (0.2-0.5): {((df['noise_contamination_rate'] > 0.2) & (df['noise_contamination_rate'] <= 0.5)).sum()} ({((df['noise_contamination_rate'] > 0.2) & (df['noise_contamination_rate'] <= 0.5)).mean()*100:.1f}%)")
    print(f"Tracks with high noise contamination (>0.5): {(df['noise_contamination_rate'] > 0.5).sum()} ({(df['noise_contamination_rate'] > 0.5).mean()*100:.1f}%)")
    print(f"Average noise points per track's kNN: {df['avg_noise_in_knn'].mean():.2f}")
    print(f"Tracks with high cluster purity (>0.8): {(df['track_cluster_purity'] > 0.8).sum()} ({(df['track_cluster_purity'] > 0.8).mean()*100:.1f}%)")
    
    summary_stats.to_csv(save_dir / 'summary_statistics.csv')

def create_summary_statistics_no_noise(df, save_dir):
    """Create a comprehensive summary of statistics including noise metrics."""
    summary_stats = df[['precision', 'recall', 'completeness', 'purity', 
                       'track_length', 'avg_pt_in_track', 'total_tp', 'total_fp', 'total_fn']].describe()
    
    print("\nSummary Statistics:")
    print("="*100)
    print(summary_stats.round(3))
    
    # Save to CSV
    summary_stats.to_csv(save_dir / 'summary_statistics.csv')

def create_noise_contamination_analysis(df, save_dir):
    """Create comprehensive noise contamination analysis plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Noise Contamination Analysis', fontsize=16, fontweight='bold')
    
    noise_contamination_bins = np.linspace(0, 1, 101)
    axes[0, 0].hist(df['noise_contamination_rate'], bins=noise_contamination_bins, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(df['noise_contamination_rate'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["noise_contamination_rate"].mean():.3f}')
    axes[0, 0].axvline(df['noise_contamination_rate'].median(), color='blue', linestyle=':', 
                       label=f'Median: {df["noise_contamination_rate"].median():.3f}')
    axes[0, 0].set_xlabel('Noise Contamination Rate')
    axes[0, 0].set_ylabel('Number of Tracks')
    axes[0, 0].set_title('Noise Contamination Rate Distribution (0-1)')
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    cluster_purity_bins = np.linspace(0, 1, 101)
    axes[0, 1].hist(df['track_cluster_purity'], bins=cluster_purity_bins, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(df['track_cluster_purity'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["track_cluster_purity"].mean():.3f}')
    axes[0, 1].axvline(df['track_cluster_purity'].median(), color='blue', linestyle=':', 
                       label=f'Median: {df["track_cluster_purity"].median():.3f}')
    axes[0, 1].set_xlabel('Track Cluster Purity (excl. noise)')
    axes[0, 1].set_ylabel('Number of Tracks')
    axes[0, 1].set_title('Track Cluster Purity Distribution (0-1)')
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    noise_knn_bins = np.linspace(0, 50, 51)
    axes[0, 2].hist(df['avg_noise_in_knn'], bins=noise_knn_bins, alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(df['avg_noise_in_knn'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["avg_noise_in_knn"].mean():.1f}')
    axes[0, 2].axvline(df['avg_noise_in_knn'].median(), color='blue', linestyle=':', 
                       label=f'Median: {df["avg_noise_in_knn"].median():.1f}')
    axes[0, 2].set_xlabel('Average Noise Points in kNN')
    axes[0, 2].set_ylabel('Number of Tracks')
    axes[0, 2].set_title('Average Noise in kNN Distribution (0-50)')
    axes[0, 2].set_xlim(0, 50)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].scatter(df['track_length'], df['noise_contamination_rate'], alpha=0.6, s=20)
    axes[1, 0].set_xlabel('Track Length')
    axes[1, 0].set_ylabel('Noise Contamination Rate')
    axes[1, 0].set_title('Noise Contamination vs Track Length')
    axes[1, 0].grid(True, alpha=0.3)
    
    corr = df['track_length'].corr(df['noise_contamination_rate'])
    axes[1, 0].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=axes[1, 0].transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    axes[1, 1].scatter(df['avg_pt_in_track'], df['noise_contamination_rate'], alpha=0.6, s=20)
    axes[1, 1].set_xlabel('Average pT in Track')
    axes[1, 1].set_ylabel('Noise Contamination Rate')
    axes[1, 1].set_title('Noise Contamination vs Average pT')
    axes[1, 1].grid(True, alpha=0.3)
    
    corr = df['avg_pt_in_track'].corr(df['noise_contamination_rate'])
    axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=axes[1, 1].transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    axes[1, 2].scatter(df['purity'], df['track_cluster_purity'], alpha=0.6, s=20)
    axes[1, 2].plot([0, 1], [0, 1], 'r--', alpha=0.8)
    axes[1, 2].set_xlabel('Regular Track Purity')
    axes[1, 2].set_ylabel('Cluster Purity (excl. noise)')
    axes[1, 2].set_title('Regular Purity vs Cluster Purity')
    axes[1, 2].grid(True, alpha=0.3)
    
    corr = df['purity'].corr(df['track_cluster_purity'])
    axes[1, 2].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=axes[1, 2].transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_dir / 'noise_contamination_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_noise_performance_by_track_length(df, save_dir):
    """Analyze noise contamination performance as a function of track length."""

    track_length_bins = [2, 3, 4, 5, 7, 10, 15, 25, max(25.1, df['track_length'].max()+1)]
    df['length_bin'] = pd.cut(df['track_length'], bins=track_length_bins, right=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Noise Contamination vs Track Length', fontsize=16, fontweight='bold')
    
    noise_metrics = ['noise_contamination_rate', 'track_cluster_purity', 'avg_noise_in_knn', 'total_noise_in_knn']
    titles = ['Noise Contamination Rate vs Track Length', 'Track Cluster Purity vs Track Length', 
              'Avg Noise in kNN vs Track Length', 'Total Noise in kNN vs Track Length']
    
    for i, (metric, title) in enumerate(zip(noise_metrics, titles)):
        ax = axes[i//2, i%2]
        
        box_data = [group[metric].values for name, group in df.groupby('length_bin', observed=False)]
        box_labels = [str(name) for name, group in df.groupby('length_bin', observed=False)]
        
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, 
                       boxprops=dict(facecolor='lightcoral', color='red', linewidth=2),
                       whiskerprops=dict(color='red', linewidth=2),
                       capprops=dict(color='red', linewidth=2),
                       medianprops=dict(color='darkblue', linewidth=2))
        
        ax.set_title(title)
        ax.set_xlabel('Track Length Range')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'noise_performance_by_track_length.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_track_vs_noise_separation_analysis(df, save_dir):
    """Analyze how well the model separates tracks from noise."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Track vs Noise Separation Analysis', fontsize=16, fontweight='bold')
    
    axes[0, 0].scatter(df['noise_contamination_rate'], df['precision'], alpha=0.6, s=20, c=df['track_length'], cmap='viridis')
    cbar1 = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
    cbar1.set_label('Track Length')
    axes[0, 0].set_xlabel('Noise Contamination Rate')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision vs Noise Contamination')
    axes[0, 0].grid(True, alpha=0.3)
    
    corr = df['noise_contamination_rate'].corr(df['precision'])
    axes[0, 0].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=axes[0, 0].transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    axes[0, 1].scatter(df['noise_contamination_rate'], df['recall'], alpha=0.6, s=20, c=df['track_length'], cmap='viridis')
    cbar2 = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
    cbar2.set_label('Track Length')
    axes[0, 1].set_xlabel('Noise Contamination Rate')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_title('Recall vs Noise Contamination')
    axes[0, 1].grid(True, alpha=0.3)
    
    corr = df['noise_contamination_rate'].corr(df['recall'])
    axes[0, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=axes[0, 1].transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    noise_levels = ['Low (0-0.2)', 'Medium (0.2-0.5)', 'High (0.5-1.0)']
    low_noise = df[df['noise_contamination_rate'] <= 0.2]
    med_noise = df[(df['noise_contamination_rate'] > 0.2) & (df['noise_contamination_rate'] <= 0.5)]
    high_noise = df[df['noise_contamination_rate'] > 0.5]
    
    noise_counts = [len(low_noise), len(med_noise), len(high_noise)]
    colors = ['green', 'orange', 'red']
    bars = axes[1, 0].bar(noise_levels, noise_counts, color=colors, alpha=0.7)
    axes[1, 0].set_ylabel('Number of Tracks')
    axes[1, 0].set_title('Track Distribution by Noise Contamination Level')
    axes[1, 0].grid(True, alpha=0.3)
    
    total_tracks = sum(noise_counts)
    for bar, count in zip(bars, noise_counts):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{count}\n({count/total_tracks*100:.1f}%)',
                        ha='center', va='bottom')
    
    noise_groups = [low_noise, med_noise, high_noise]
    metrics = ['precision', 'recall', 'completeness', 'track_cluster_purity']
    metric_labels = ['Precision', 'Recall', 'Completeness', 'Cluster Purity']
    
    x = np.arange(len(noise_levels))
    width = 0.2
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        means = [group[metric].mean() if len(group) > 0 else 0 for group in noise_groups]
        axes[1, 1].bar(x + i*width, means, width, label=label, alpha=0.7)
    
    axes[1, 1].set_xlabel('Noise Contamination Level')
    axes[1, 1].set_ylabel('Performance Metric')
    axes[1, 1].set_title('Performance by Noise Contamination Level')
    axes[1, 1].set_xticks(x + width * 1.5)
    axes[1, 1].set_xticklabels(noise_levels)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'track_vs_noise_separation.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)