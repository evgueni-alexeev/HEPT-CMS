import torch
from pathlib import Path


def calculate_cluster_connectivity(cluster_indices, nearby_point_pairs):
    """
    Calculate what percentage of points in a cluster are connected to each other.
    Returns a float between 0 and 1 representing the connectivity percentage.
    """
    if len(cluster_indices) <= 1:
        return 0.0
        
    # Get all edges between points in this cluster
    cluster_edges = nearby_point_pairs[:, torch.isin(nearby_point_pairs[0], cluster_indices) & 
                                           torch.isin(nearby_point_pairs[1], cluster_indices)]
    
    # Initialize union-find data structures
    parent = {idx.item(): idx.item() for idx in cluster_indices}
    size = {idx.item(): 1 for idx in cluster_indices}
    
    def find_set(x):
        if parent[x] != x:
            parent[x] = find_set(parent[x])
        return parent[x]
    
    def union_set(a, b):
        ra = find_set(a)
        rb = find_set(b)
        if ra != rb:
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
    
    # Union all edges
    for i in range(cluster_edges.size(1)):
        u, v = cluster_edges[0, i].item(), cluster_edges[1, i].item()
        union_set(u, v)
    
    # Find the largest connected component
    component_sizes = {}
    for idx in cluster_indices:
        root = find_set(idx.item())
        component_sizes[root] = component_sizes.get(root, 0) + 1
    
    if not component_sizes:
        return 0.0
        
    largest_component_size = max(component_sizes.values())
    return largest_component_size / len(cluster_indices)


def calculate_cluster_noise_contamination(cluster_indices, nearby_point_pairs, cluster_ids):
    """Return (noise_count, total_neighbors, signal_points) for this cluster."""
    nbr_idx = nearby_point_pairs[1][torch.isin(nearby_point_pairs[0], cluster_indices)].unique()
    if nbr_idx.numel() == 0:
        return 0, 0, len(cluster_indices)  # no neighbors, but still count signal points
    
    noise_mask = cluster_ids[nbr_idx] < 0
    noise_count = int(noise_mask.sum().item())
    total_neighbors = int(nbr_idx.numel())
    signal_points = len(cluster_indices)
    
    return noise_count, total_neighbors, signal_points


def save_cluster_noise_contamination(noise_metrics, parent_dir, event_id):
    """Save noise contamination metrics to combined cluster analysis files.
    
    noise_metrics: list of (cluster_id, noise_count, total_neighbors, signal_points) tuples
    """
    event_id = int(event_id)
    
    # Extract counts for this event
    total_noise_points = sum(noise_count for _, noise_count, _, _ in noise_metrics)
    total_signal_points = sum(signal_pts for _, _, _, signal_pts in noise_metrics)
    total_neighbor_checks = sum(total_nbr for _, _, total_nbr, _ in noise_metrics)
    
    # Calculate averages
    avg_noise_ratio = total_noise_points / total_neighbor_checks if total_neighbor_checks > 0 else 0.0
    signal_to_noise_ratio = total_signal_points / total_noise_points if total_noise_points > 0 else float('inf')
    
    # Store noise data for combined summary (will be written by save_cluster_connectivity)
    noise_data = {
        'noise_pts': total_noise_points,
        'signal_pts': total_signal_points,
        'SNR': signal_to_noise_ratio,
        'avg_noise_ratio': avg_noise_ratio,
        'num_clusters': len(noise_metrics)
    }
    
    # Store in a temporary attribute for the connectivity function to access
    if not hasattr(save_cluster_noise_contamination, 'pending_noise_data'):
        save_cluster_noise_contamination.pending_noise_data = {}
    save_cluster_noise_contamination.pending_noise_data[event_id] = noise_data
    
    # Update global noise stats
    stats_file = parent_dir / "cluster_analysis_stats.txt"
    
    # Read current totals for noise
    try:
        with open(stats_file, "r") as f:
            lines = f.readlines()
            if len(lines) > 1:
                last = lines[-1].strip().split(',')
                if len(last) >= 5:  # New combined format
                    tot_noise_pts = int(last[3])
                    tot_signal_pts = int(last[4])
                else:
                    tot_noise_pts = tot_signal_pts = 0
            else:
                tot_noise_pts = tot_signal_pts = 0
    except (FileNotFoundError, ValueError, IndexError):
        tot_noise_pts = tot_signal_pts = 0
    
    # Update running totals
    tot_noise_pts += total_noise_points
    tot_signal_pts += total_signal_points
    
    overall_snr = tot_signal_pts / tot_noise_pts if tot_noise_pts > 0 else float('inf')
    # noise_contamination_ratio = fraction of all neighbor points that are noise (0=no noise, 1=all noise)
    overall_noise_ratio = tot_noise_pts / (tot_noise_pts + tot_signal_pts) if (tot_noise_pts + tot_signal_pts) > 0 else 0.0
    
    # Store updated noise stats for connectivity function to use
    save_cluster_noise_contamination.global_noise_stats = {
        'tot_noise_pts': tot_noise_pts,
        'tot_signal_pts': tot_signal_pts,
        'overall_snr': overall_snr,
        'overall_noise_ratio': overall_noise_ratio
    }


def save_cluster_connectivity(connectivity_metrics, parent_dir, event_id):
    event_id = int(event_id)
    metrics_file = parent_dir / f"event_{event_id}.txt"
    
    # Check if metrics include track lengths (3-tuple) or just connectivity (2-tuple)
    has_track_lengths = len(connectivity_metrics) > 0 and len(connectivity_metrics[0]) == 3
    
    # Calculate connectivity statistics
    if has_track_lengths:
        connectivities = [conn for _, conn, _ in connectivity_metrics]
        track_lengths = [tlen for _, _, tlen in connectivity_metrics]
        avg_connectivity = sum(connectivities) / len(connectivities) if connectivities else 0.0
        avg_track_length = sum(track_lengths) / len(track_lengths) if track_lengths else 0.0
    else:
        connectivities = [conn for _, conn in connectivity_metrics]
        avg_connectivity = sum(connectivities) / len(connectivities) if connectivities else 0.0
        avg_track_length = 0.0
    
    # Get noise data if available
    noise_data = None
    if hasattr(save_cluster_noise_contamination, 'pending_noise_data'):
        noise_data = save_cluster_noise_contamination.pending_noise_data.get(event_id)
    
    # Write detailed metrics to individual event file
    with open(metrics_file, "w") as f:
        f.write(f"Event {event_id} Statistics:\n")
        f.write(f"Number of clusters: {len(connectivity_metrics)}\n")
        f.write(f"Average connectivity: {avg_connectivity:.4f}\n")
        if has_track_lengths:
            f.write(f"Average track length: {avg_track_length:.2f}\n")
        
        if noise_data:
            f.write(f"Noise contamination: {noise_data['noise_pts']} noise points, {noise_data['signal_pts']} signal points\n")
            f.write(f"Signal-to-noise ratio: {noise_data['SNR']:.2f}\n")
            f.write(f"Average noise ratio: {noise_data['avg_noise_ratio']:.4f}\n")
        
        f.write(f"\nDetailed Cluster Metrics:\n")
        if has_track_lengths:
            f.write(f"{'Cluster ID':<12} {'Connectivity':<12} {'Track Length':<12}\n")
            f.write("-" * 40 + "\n")
            for cluster_id, connectivity, track_length in connectivity_metrics:
                f.write(f"{cluster_id:<12} {connectivity:<12.4f} {track_length:<12}\n")
        else:
            f.write(f"{'Cluster ID':<12} {'Connectivity':<12}\n")
            f.write("-" * 26 + "\n")
            for cluster_id, connectivity in connectivity_metrics:
                f.write(f"{cluster_id:<12} {connectivity:<12.4f}\n")
    
    # Write combined summary
    summary_file = parent_dir / "cluster_analysis_summary.txt"
    with open(summary_file, "a") as f:
        summary_line = f"{event_id}: connectivity={avg_connectivity:.4f}"
        if has_track_lengths:
            summary_line += f", track_len={avg_track_length:.2f}"
        if noise_data:
            summary_line += f", noise_pts={noise_data['noise_pts']}, signal_pts={noise_data['signal_pts']}, SNR={noise_data['SNR']:.2f}"
        summary_line += f" ({len(connectivity_metrics)} clusters)\n"
        f.write(summary_line)
    
    # Update combined global statistics
    stats_file = parent_dir / "cluster_analysis_stats.txt"
    
    # Read current connectivity stats
    try:
        with open(stats_file, "r") as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip().split(",")
                if len(last_line) >= 3:  # New format
                    total_clusters = int(last_line[0])
                    total_connectivity = float(last_line[1]) * total_clusters  # Convert back from weighted avg
                    total_track_length = float(last_line[2]) * total_clusters  # Convert back from avg
                else:
                    total_clusters = 0
                    total_connectivity = 0.0
                    total_track_length = 0.0
            else:
                total_clusters = 0
                total_connectivity = 0.0
                total_track_length = 0.0
    except (FileNotFoundError, IndexError, ValueError):
        total_clusters = 0
        total_connectivity = 0.0
        total_track_length = 0.0
    
    # Update connectivity stats
    total_clusters += len(connectivity_metrics)
    total_connectivity += sum(connectivities)
    if has_track_lengths:
        total_track_length += sum(track_lengths)
    
    weighted_avg_connectivity = total_connectivity / total_clusters if total_clusters > 0 else 0.0
    avg_track_length_overall = total_track_length / total_clusters if total_clusters > 0 else 0.0
    
    # Get noise stats if available
    noise_stats = getattr(save_cluster_noise_contamination, 'global_noise_stats', {
        'tot_noise_pts': 0,
        'tot_signal_pts': 0,
        'overall_snr': float('inf'),
        'overall_noise_ratio': 0.0
    })
    
    # Write combined stats file
    if not stats_file.exists():
        with open(stats_file, "w") as f:
            f.write("total_clusters,weighted_avg_connectivity,avg_track_length,noise_points,signal_points,overall_SNR,noise_contamination_ratio\n")
    
    with open(stats_file, "w") as f:
        f.write("total_clusters,weighted_avg_connectivity,avg_track_length,noise_points,signal_points,overall_SNR,noise_contamination_ratio\n")
        f.write(f"{total_clusters},{weighted_avg_connectivity:.4f},{avg_track_length_overall:.2f},"
                f"{noise_stats['tot_noise_pts']},{noise_stats['tot_signal_pts']},"
                f"{noise_stats['overall_snr']:.4f},{noise_stats['overall_noise_ratio']:.4f}\n")
    
    # Clean up noise data after processing
    if hasattr(save_cluster_noise_contamination, 'pending_noise_data') and event_id in save_cluster_noise_contamination.pending_noise_data:
        del save_cluster_noise_contamination.pending_noise_data[event_id] 