import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[1]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import torch
import yaml
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import argparse

from filter_model import EventDataset, _load_model

VISUALS = 1
PILEUP_CKPT = "pu200_0.409p"
TRACKING_CKPT = "pu200_0.991AP"
RECALL = 0.99
NUM_EVENTS = 5
SPLIT = "all"  # "test" or "all"
USE_ONLY_TRUE_TRACKS = True

def calculate_efficiency(embeddings, particle_ids, mask, dist_metric, pt_values, eta_values):
    """
    Calculate efficiency using double-majority (DM), LHC, and perfect clustering criteria.
    
    DM definition in this case is:
      For each track, check two conditions:
        1. >50% of track's points have >50% of their K nearest neighbors belong to the correct track
        2. Mutual connectivity: >50% of track's points are in each other's K nearest neighbors

    """
    # Apply mask
    masked_embeddings = embeddings[mask]
    masked_particle_ids = particle_ids[mask]
    masked_pt_values = pt_values[mask]
    masked_eta_values = eta_values[mask]
    
    if len(masked_embeddings) == 0:
        empty_result = {'efficiency': 0.0, 'matched_tracks': 0, 'total_tracks': 0}
        return {
            'dm_efficiency': {
                'all_data': empty_result,
                'high_pt': empty_result,
                'high_pt_3plus': empty_result,
                'high_pt_4plus': empty_result,
            },
            'lhc_efficiency': {
                'all_data': empty_result,
                'high_pt': empty_result,
                'high_pt_3plus': empty_result,
                'high_pt_4plus': empty_result,
            },
            'perfect_efficiency': {
                'all_data': empty_result,
                'high_pt': empty_result,
                'high_pt_3plus': empty_result,
                'high_pt_4plus': empty_result,
            }
        }
    
    # Separate track points from noise
    track_mask = (masked_particle_ids != -1)
    track_embeddings = masked_embeddings[track_mask]
    track_particle_ids = masked_particle_ids[track_mask]
    track_pt_values = masked_pt_values[track_mask]
    track_eta_values = masked_eta_values[track_mask]
    
    # Get unique tracks (excluding background -1)
    unique_tracks = torch.unique(track_particle_ids)
    
    # Double-majority efficiency counters
    dm_counters = {
        'all_data': {'total': 0, 'matched': 0},
        'high_pt': {'total': 0, 'matched': 0},  # pt > 0.9
        'high_pt_3plus': {'total': 0, 'matched': 0},  # pt > 0.9 + size >= 3
        'high_pt_4plus': {'total': 0, 'matched': 0},  # pt > 0.9 + size >= 4
    }
    
    # LHC efficiency counters
    lhc_counters = {
        'all_data': {'total': 0, 'matched': 0},
        'high_pt': {'total': 0, 'matched': 0},
        'high_pt_3plus': {'total': 0, 'matched': 0},
        'high_pt_4plus': {'total': 0, 'matched': 0},
    }
    
    # Perfect clustering efficiency counters
    perfect_counters = {
        'all_data': {'total': 0, 'matched': 0},
        'high_pt': {'total': 0, 'matched': 0},
        'high_pt_3plus': {'total': 0, 'matched': 0},
        'high_pt_4plus': {'total': 0, 'matched': 0},
    }
    
    for track_id in unique_tracks:
        track_mask_specific = (track_particle_ids == track_id)
        track_indices = track_mask_specific.nonzero().flatten()
        track_size = len(track_indices)
        
        if track_size < 2:  # Skip single-point tracks
            continue
            
        track_embs = track_embeddings[track_indices]
        track_pts = track_pt_values[track_indices]
        track_etas = track_eta_values[track_indices]
        
        # Check track quality categories (using track averages)
        avg_pt = track_pts.mean().item()
        is_high_pt = (avg_pt > 0.9)
        is_high_pt_3plus = is_high_pt and (track_size >= 3)
        is_high_pt_4plus = is_high_pt and (track_size >= 4)
        
        # Calculate distances within this track and to all other points
        if "l2" in dist_metric:
            # Distance from track points to all track points (including itself)
            intra_track_dist = torch.cdist(track_embs, track_embeddings, p=2.0)
            # Distance from track points to all masked points (for neighborhood analysis)
            all_points_dist = torch.cdist(track_embs, masked_embeddings, p=2.0)
        elif dist_metric == "cosine":
            intra_track_dist = 1 - F.cosine_similarity(track_embs.unsqueeze(1), track_embeddings.unsqueeze(0), dim=-1)
            all_points_dist = 1 - F.cosine_similarity(track_embs.unsqueeze(1), masked_embeddings.unsqueeze(0), dim=-1)
        
        # Set K for analysis - exactly track_size - 1 (all other points in the track)
        k = track_size - 1
        
        # Condition 1: For each point, check if >=50% of its K nearest neighbors belong to correct track
        # LHC: For each point, check if >=75% of its K nearest neighbors belong to correct track  
        # Perfect: For each point, check if ALL K nearest neighbors belong to correct track
        points_with_good_purity = 0
        points_with_lhc_purity = 0
        points_with_perfect_purity = 0
        
        for i in range(track_size):
            # Find K nearest neighbors among all points
            distances = all_points_dist[i]
            _, nearest_indices = torch.topk(distances, k + 1, largest=False)  # +1 to exclude self
            nearest_indices = nearest_indices[1:]  # Remove self
            
            # Check how many belong to correct track
            neighbor_particle_ids = masked_particle_ids[nearest_indices]
            correct_neighbors = (neighbor_particle_ids == track_id).sum().item()
            
            if correct_neighbors / k >= 0.5:  # >=50% purity for DM
                points_with_good_purity += 1
            
            if correct_neighbors / k >= 0.75:  # >=75% purity for LHC
                points_with_lhc_purity += 1
                
            if correct_neighbors == k:  # 100% purity for Perfect clustering
                points_with_perfect_purity += 1
        
        condition1_met = (points_with_good_purity / track_size) >= 0.5
        lhc_condition_met = (points_with_lhc_purity / track_size) >= 0.75
        perfect_condition_met = (points_with_perfect_purity == track_size)  # ALL points must have perfect purity
        
        # Condition 2: Mutual connectivity within track
        # For each point, check if >50% of other track points are in its K nearest neighbors
        points_with_good_connectivity = 0
        
        for i in range(track_size):
            # Find K nearest neighbors among all track points
            distances = intra_track_dist[i]
            _, nearest_indices = torch.topk(distances, min(k + 1, track_size), largest=False)
            nearest_indices = nearest_indices[1:]  # Remove self
            
            # Count how many other track points are in the neighborhood
            # (All neighbors are from the same track by construction)
            neighbors_found = len(nearest_indices)
            other_track_points = track_size - 1  # Exclude self
            
            if other_track_points > 0 and (neighbors_found / other_track_points) >= 0.5:
                points_with_good_connectivity += 1
        
        condition2_met = (points_with_good_connectivity / track_size) >= 0.5
        
        # Track matches if both conditions are met
        dm_track_matched = condition1_met and condition2_met
        # LHC efficiency: only requires >=75% neighbor purity condition
        lhc_track_matched = lhc_condition_met
        # Perfect clustering: requires 100% neighbor purity for all points
        perfect_track_matched = perfect_condition_met
        
        # Update counters for all categories
        categories = [
            ('all_data', True),
            ('high_pt', is_high_pt),
            ('high_pt_3plus', is_high_pt_3plus),
            ('high_pt_4plus', is_high_pt_4plus),
        ]
        
        for category, condition in categories:
            if condition:
                # Double-majority counters
                dm_counters[category]['total'] += 1
                if dm_track_matched:
                    dm_counters[category]['matched'] += 1
                    
                # LHC efficiency counters
                lhc_counters[category]['total'] += 1
                if lhc_track_matched:
                    lhc_counters[category]['matched'] += 1
                    
                # Perfect clustering counters
                perfect_counters[category]['total'] += 1
                if perfect_track_matched:
                    perfect_counters[category]['matched'] += 1
    
    # Calculate efficiencies for all categories
    def calculate_efficiency(counters):
        return {
            category: {
                'efficiency': data['matched'] / data['total'] if data['total'] > 0 else 0.0,
                'matched_tracks': data['matched'],
                'total_tracks': data['total']
            }
            for category, data in counters.items()
        }
    
    return {
        'dm_efficiency': calculate_efficiency(dm_counters),
        'lhc_efficiency': calculate_efficiency(lhc_counters),
        'perfect_efficiency': calculate_efficiency(perfect_counters)
    }

def analyze_track_performance(embeddings, particle_ids, mask, dist_metric, pt_threshold, event_idx, pt_values):
    """
    Analyze model performance on a track-by-track basis, including noise contamination analysis.
    """
    masked_embeddings = embeddings[mask]
    masked_particle_ids = particle_ids[mask]
    masked_pt_values = pt_values[mask]
    
    if len(masked_embeddings) == 0:
        return [], {}
    
    # Separate track points from noise points
    track_mask = (masked_particle_ids != -1)
    noise_mask = (masked_particle_ids == -1)
    
    track_embeddings = masked_embeddings[track_mask]
    track_particle_ids = masked_particle_ids[track_mask]
    track_pt_values = masked_pt_values[track_mask]
    
    noise_embeddings = masked_embeddings[noise_mask]
    noise_indices = noise_mask.nonzero().flatten()
    
    # Get unique tracks (excluding background -1)
    unique_tracks = torch.unique(track_particle_ids)
    
    track_results = []
    noise_analysis = {
        'total_noise_points': len(noise_embeddings),
        'total_track_points': len(track_embeddings),
        'noise_contamination_by_track': [],
        'avg_noise_contamination': 0.0,
        'noise_to_track_distances': [],
        'track_cluster_purity': [],
        'noise_rejection_rate': 0.0
    }
    
    all_noise_contaminations = []
    all_track_purities = []
    
    for track_id in unique_tracks:
        track_mask_specific = (track_particle_ids == track_id)
        track_indices = track_mask_specific.nonzero().flatten()
        track_size = len(track_indices)
        
        if track_size < 2:  # Skip single-point tracks
            continue
            
        track_embs = track_embeddings[track_indices]
        track_pts = track_pt_values[track_indices]
        avg_pt = track_pts.mean().item()
        
        # Calculate distances from each track point to ALL points (tracks + noise)
        if "l2" in dist_metric:
            dist_matrix = torch.cdist(track_embs, masked_embeddings, p=2.0)
        elif dist_metric == "cosine":
            dist_matrix = 1 - F.cosine_similarity(track_embs.unsqueeze(1), masked_embeddings.unsqueeze(0), dim=-1)
        
        # For each point in this track, find its nearest neighbors and analyze noise contamination
        track_performance = {
            'event_idx': event_idx,
            'track_id': track_id.item(),
            'track_size': track_size,
            'avg_pt': avg_pt,
            'points_analyzed': [],
            'track_completeness': 0.0,  # fraction of track points found by model
            'track_purity': 0.0,        # fraction of predicted points that are correct
            'avg_precision': 0.0,
            'avg_recall': 0.0,
            'noise_contamination_rate': 0.0,  # fraction of kNN that are noise
            'track_cluster_purity': 0.0,      # purity considering noise
            'avg_noise_in_knn': 0.0          # average number of noise points in kNN
        }
        
        total_precision = 0.0
        total_recall = 0.0
        correctly_found_points = 0
        total_noise_contamination = 0.0
        
        for i, point_idx in enumerate(track_indices):
            # Get k nearest neighbors - exactly track_size - 1 (all other points in the track)
            k = track_size - 1
            distances = dist_matrix[i]
            _, nearest_indices = torch.topk(distances, k + 1, largest=False)  # +1 to exclude self
            nearest_indices = nearest_indices[1:]  # Remove self
            
            # Get predicted track members (nearest neighbors)
            predicted_particle_ids = masked_particle_ids[nearest_indices]
            
            # Analyze noise contamination in kNN
            noise_in_knn = (predicted_particle_ids == -1).sum().item()
            track_points_in_knn = (predicted_particle_ids == track_id).sum().item()
            other_tracks_in_knn = ((predicted_particle_ids != -1) & (predicted_particle_ids != track_id)).sum().item()
            
            # Calculate metrics
            true_positives = track_points_in_knn
            false_positives = noise_in_knn + other_tracks_in_knn
            false_negatives = track_size - 1 - true_positives
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (track_size - 1) if (track_size - 1) > 0 else 1.0
            
            # Noise contamination metrics
            noise_contamination = noise_in_knn / k if k > 0 else 0.0
            cluster_purity = true_positives / k if k > 0 else 0.0
            
            total_noise_contamination += noise_contamination
            
            # Check if this point found most of its track
            if recall > 0.5:
                correctly_found_points += 1
            
            point_result = {
                'point_idx_in_track': i,
                'global_point_idx': point_idx.item(),
                'k_neighbors': k,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'precision': precision,
                'recall': recall,
                'noise_in_knn': noise_in_knn,
                'other_tracks_in_knn': other_tracks_in_knn,
                'noise_contamination_rate': noise_contamination,
                'cluster_purity': cluster_purity,
                'predicted_particle_ids': predicted_particle_ids.cpu().tolist(),
                'expected_track_id': track_id.item()
            }
            
            track_performance['points_analyzed'].append(point_result)
            total_precision += precision
            total_recall += recall
        
        # Calculate track-level metrics
        track_performance['track_completeness'] = correctly_found_points / track_size
        track_performance['avg_precision'] = total_precision / track_size
        track_performance['avg_recall'] = total_recall / track_size
        track_performance['noise_contamination_rate'] = total_noise_contamination / track_size
        track_performance['avg_noise_in_knn'] = sum(p['noise_in_knn'] for p in track_performance['points_analyzed']) / track_size
        
        # Calculate overall track purity considering noise
        all_predictions = []
        for point_data in track_performance['points_analyzed']:
            all_predictions.extend(point_data['predicted_particle_ids'])
        
        if all_predictions:
            correct_predictions = sum(1 for pid in all_predictions if pid == track_id.item())
            track_performance['track_purity'] = correct_predictions / len(all_predictions)
            # Track cluster purity (excluding noise from denominator)
            non_noise_predictions = sum(1 for pid in all_predictions if pid != -1)
            track_performance['track_cluster_purity'] = correct_predictions / non_noise_predictions if non_noise_predictions > 0 else 0.0
        
        track_results.append(track_performance)
        all_noise_contaminations.append(track_performance['noise_contamination_rate'])
        all_track_purities.append(track_performance['track_cluster_purity'])
        
        # Add to noise analysis
        noise_analysis['noise_contamination_by_track'].append({
            'track_id': track_id.item(),
            'track_size': track_size,
            'avg_pt': avg_pt,
            'noise_contamination_rate': track_performance['noise_contamination_rate'],
            'track_cluster_purity': track_performance['track_cluster_purity']
        })
    
    # Global noise analysis
    if all_noise_contaminations:
        noise_analysis['avg_noise_contamination'] = np.mean(all_noise_contaminations)
        noise_analysis['track_cluster_purity'] = all_track_purities
    
    # Analyze noise point behavior in embedding space
    if len(noise_embeddings) > 0 and len(track_embeddings) > 0:
        if "l2" in dist_metric:
            noise_to_track_dist = torch.cdist(noise_embeddings, track_embeddings, p=2.0)
        elif dist_metric == "cosine":
            noise_to_track_dist = 1 - F.cosine_similarity(noise_embeddings.unsqueeze(1), track_embeddings.unsqueeze(0), dim=-1)
        
        min_distances, _ = torch.min(noise_to_track_dist, dim=1)
        noise_analysis['noise_to_track_distances'] = min_distances.cpu().numpy().tolist()
        noise_analysis['avg_noise_to_track_distance'] = min_distances.mean().item()
        noise_analysis['noise_separation_quality'] = min_distances.std().item()  # Higher std means better separation
    
    return track_results, noise_analysis

def print_efficiency(dm_results, lhc_results, perfect_results):
    """Print double-majority, LHC, and perfect clustering efficiency results."""
    print(f"\n{'='*80}")
    print(f"TRACKING EFFICIENCY ANALYSIS")
    print(f"{'='*80}")
    print("Double-Majority: ≥50% purity + ≥50% connectivity")
    print("LHC Efficiency: ≥75% purity")
    print("Perfect Clustering: 100% purity (all k neighbors from same track)")
    print()
    
    def aggregate_results(results_dict):
        """Aggregate results across all events for a given efficiency type."""
        aggregated = {}
        for category, event_results in results_dict.items():
            total_matched = sum(r['matched_tracks'] for r in event_results)
            total_tracks = sum(r['total_tracks'] for r in event_results)
            efficiency = total_matched / total_tracks if total_tracks > 0 else 0.0
            aggregated[category] = {
                'efficiency': efficiency,
                'matched_tracks': total_matched,
                'total_tracks': total_tracks
            }
        return aggregated
    
    dm_agg = aggregate_results(dm_results)
    lhc_agg = aggregate_results(lhc_results)
    perfect_agg = aggregate_results(perfect_results)
    
    # Print results for all categories
    categories = [
        ('all_data', 'All tracks'),
        ('high_pt', 'High-pt (pt>0.9)'),
        ('high_pt_3plus', 'High-pt 3+ (pt>0.9, size≥3)'),
        ('high_pt_4plus', 'High-pt 4+ (pt>0.9, size≥4)'),
    ]
    
    for category, label in categories:
        dm_data = dm_agg[category]
        lhc_data = lhc_agg[category]
        perfect_data = perfect_agg[category]
        
        print(f"{label}:")
        print(f"  DM:      {dm_data['matched_tracks']}/{dm_data['total_tracks']} = {dm_data['efficiency']:.3f} ({dm_data['efficiency']*100:.1f}%)")
        print(f"  LHC:     {lhc_data['matched_tracks']}/{lhc_data['total_tracks']} = {lhc_data['efficiency']:.3f} ({lhc_data['efficiency']*100:.1f}%)")
        print(f"  Perfect: {perfect_data['matched_tracks']}/{perfect_data['total_tracks']} = {perfect_data['efficiency']:.3f} ({perfect_data['efficiency']*100:.1f}%)")
        print()
    
    print()

def print_track_summary(track_results, pt_threshold):
    """Print a summary of track analysis results including noise contamination."""
    if not track_results:
        print(f"No tracks found for pt_threshold = {pt_threshold}")
        return
    
    print(f"\n{'='*80}")
    print(f"TRACK-BY-TRACK ANALYSIS (pt_threshold = {pt_threshold})")
    print(f"{'='*80}")
    
    total_tracks = len(track_results)
    avg_completeness = np.mean([t['track_completeness'] for t in track_results])
    avg_purity = np.mean([t['track_purity'] for t in track_results])
    avg_precision = np.mean([t['avg_precision'] for t in track_results])
    avg_recall = np.mean([t['avg_recall'] for t in track_results])
    
    avg_noise_contamination = np.mean([t['noise_contamination_rate'] for t in track_results])
    avg_cluster_purity = np.mean([t['track_cluster_purity'] for t in track_results])
    avg_noise_in_knn = np.mean([t['avg_noise_in_knn'] for t in track_results])
    
    print(f"Total tracks analyzed: {total_tracks}")
    print(f"Average track completeness: {avg_completeness:.3f}")
    print(f"Average track purity: {avg_purity:.3f}")
    print(f"Average precision: {avg_precision:.3f}")
    print(f"Average recall: {avg_recall:.3f}")
    if not USE_ONLY_TRUE_TRACKS:
        print(f"\nNOISE CONTAMINATION ANALYSIS:")
        print(f"Average noise contamination rate: {avg_noise_contamination:.3f}")
        print(f"Average track cluster purity (excl. noise): {avg_cluster_purity:.3f}")
        print(f"Average noise points in kNN: {avg_noise_in_knn:.1f}")
    
    track_sizes = [t['track_size'] for t in track_results]
    print(f"\nTrack size distribution:")
    print(f"  Min: {min(track_sizes)}, Max: {max(track_sizes)}, Mean: {np.mean(track_sizes):.1f}")
    
    size_bins = [(2, 4), (5, 9), (10, 19), (20, float('inf'))]
    for min_size, max_size in size_bins:
        bin_tracks = [t for t in track_results if min_size <= t['track_size'] <= max_size]
        if bin_tracks:
            bin_completeness = np.mean([t['track_completeness'] for t in bin_tracks])
            bin_purity = np.mean([t['track_purity'] for t in bin_tracks])
            bin_noise_contamination = np.mean([t['noise_contamination_rate'] for t in bin_tracks])
            bin_cluster_purity = np.mean([t['track_cluster_purity'] for t in bin_tracks])
            size_label = f"{min_size}-{max_size if max_size != float('inf') else '∞'}"
            print(f"  Size {size_label}: {len(bin_tracks)} tracks, completeness={bin_completeness:.3f}, "
                  f"purity={bin_purity:.3f}" + (f", noise_contamination={bin_noise_contamination:.3f}, cluster_purity={bin_cluster_purity:.3f}" if not USE_ONLY_TRUE_TRACKS else ""))

def save_track_results_to_csv(all_track_results, output_file):
    csv_data = []
    
    all_tracks = []
    for pt_threshold, track_results in all_track_results.items():
        all_tracks.extend(track_results)
    
    unique_tracks = {}
    for track in all_tracks:
        track_key = (track['event_idx'], track['track_id'])
        if track_key not in unique_tracks:
            unique_tracks[track_key] = track
    
    for track in unique_tracks.values():
        total_tp = sum(point['true_positives'] for point in track['points_analyzed'])
        total_fp = sum(point['false_positives'] for point in track['points_analyzed'])
        total_fn = sum(point['false_negatives'] for point in track['points_analyzed'])
        
        total_noise_in_knn = sum(point['noise_in_knn'] for point in track['points_analyzed'])
        total_other_tracks_in_knn = sum(point['other_tracks_in_knn'] for point in track['points_analyzed'])
        avg_cluster_purity = np.mean([point['cluster_purity'] for point in track['points_analyzed']])
        
        track_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        track_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        
        csv_row = {
            'event_idx': int(track['event_idx']) if not isinstance(track['event_idx'], torch.Tensor) else int(track['event_idx'].item()),
            'unique_track_id': track['track_id'],
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'average_tp': total_tp/(track['track_size']-1),
            'average_fp': total_fp/(track['track_size']-1),
            'average_fn': total_fn/(track['track_size']-1),
            'precision': track_precision,
            'recall': track_recall,
            'completeness': track['track_completeness'],
            'purity': track['track_purity'],
            'track_length': track['track_size'],
            'avg_pt_in_track': track['avg_pt'],
        }
        csv_data.append(csv_row)
        if not USE_ONLY_TRUE_TRACKS:
            csv_row.update({
            'noise_contamination_rate': track['noise_contamination_rate'],
            'track_cluster_purity': track['track_cluster_purity'],
            'avg_noise_in_knn': track['avg_noise_in_knn'],
            'total_noise_in_knn': total_noise_in_knn,
            'total_other_tracks_in_knn': total_other_tracks_in_knn,
            'avg_point_cluster_purity': avg_cluster_purity
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    
    return df

def generate_track_visualizations(output_dir, csv_file_path):
    print('Generating track visualizations...')
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    viz_path = Path(__file__).parent / "tracking/visualize_track_performance.py"
    assert viz_path.exists(), f"Visualization script not found: {viz_path}"

    from tracking.visualize_track_performance import (
        load_and_clean_data, create_performance_overview, create_performance_by_track_length, create_performance_by_pt, identify_problem_tracks, create_summary_statistics,
        create_noise_contamination_analysis, create_noise_performance_by_track_length, create_summary_statistics_no_noise
    )
            
    df = load_and_clean_data(str(csv_file_path))
        
    if USE_ONLY_TRUE_TRACKS:
        print("  - Skipping noise analysis (only true LS included)")

    print("  - Creating performance overview...")
    create_performance_overview(df, output_dir)
    
    print("  - Analyzing performance by track length...")
    perf_by_length = create_performance_by_track_length(df, output_dir)
    
    print("  - Analyzing performance by average pT...")
    create_performance_by_pt(df, output_dir)

    print("  - Identifying problem tracks...")
    identify_problem_tracks(df, output_dir)

    print("\nPerformance Statistics by Track Length:")
    print("="*60)
    print(perf_by_length)
    
    if not USE_ONLY_TRUE_TRACKS:
        create_summary_statistics(df, output_dir)
        
        print("\nNoise Analysis:")
        print("  - Creating noise contamination analysis...")
        create_noise_contamination_analysis(df, output_dir)
        
        print("  - Analyzing noise performance by track length...")
        create_noise_performance_by_track_length(df, output_dir)
        
    else:
        create_summary_statistics_no_noise(df, output_dir)
        
def main():
    parser = argparse.ArgumentParser(description="Run tracking evaluation on filtered pile-up dataset")
    parser.add_argument("-cp", "--pileup-ckpt", default=PILEUP_CKPT, help="Pile-up model ckpt folder (under eval/pileup)")
    parser.add_argument("-ct", "--tracking-ckpt", default=TRACKING_CKPT, help="Tracking model ckpt folder (under eval/tracking)")
    parser.add_argument("-r", "--recall-suffix", type=int, default=RECALL, help="Recall suffix used in filtered file name e.g. 990 → filtered_r990_…")
    parser.add_argument("-n", "--num-events", type=int, default=NUM_EVENTS, help="Number of events used when filtering (data-N.pt)")
    parser.add_argument("-s", "--split", choices=["test", "all"], default=SPLIT, help="Evaluate only test split or the full dataset")
    args = parser.parse_args()

    filtered_dir = Path(__file__).parent / f"pileup/{args.pileup_ckpt}/filtered_data"
    filtered_file = filtered_dir / f"filtered_r{int(1000*args.recall_suffix)}_data-{args.num_events}.pt"
    assert filtered_file.exists(), f"Filtered dataset not found: {filtered_file}"

    root_dir = Path(__file__).parent / f"tracking/{args.tracking_ckpt}"
    ckpt_dir = root_dir / "best.ckpt"
    yaml_dir = root_dir / "hparams.yaml"

    output_dir = root_dir / "track_analysis" / f"N={args.num_events}_r={args.recall_suffix}_{'filtered_LS' if not USE_ONLY_TRUE_TRACKS else 'only_true_LS'}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data, slices, idx_split = torch.load(filtered_file, weights_only=False)
    selected_dataset = EventDataset(data, slices)
    selected_dataset.idx_split = idx_split

    if args.split == "test":
        dataset = selected_dataset.index_select(idx_split["test"])
    else:
        dataset = selected_dataset

    sample = selected_dataset[0]
    in_dim = sample.x.shape[1]
    coords_dim = sample.coords.shape[1]

    model = _load_model(ckpt_path=ckpt_dir, hparams_path=yaml_dir, in_dim=in_dim, coords_dim=coords_dim, task="tracking")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define pt thresholds and distance metric here
    pt_thres = [0]
    dist_metric = ['l2']
    
    all_metrics = {pt: {'accuracy': [], 'precision': [], 'recall': []} for pt in pt_thres}
    all_track_results = {pt: [] for pt in pt_thres}
    
    all_dm_efficiency = {
        'all_data': [], 
        'high_pt': [],        # pt>0.9 (using track averages)
        'high_pt_3plus': [],  # pt>0.9 + track size >= 3
        'high_pt_4plus': [],  # pt>0.9 + track size >= 4
    }

    all_lhc_efficiency = {
        'all_data': [],
        'high_pt': [],
        'high_pt_3plus': [],
        'high_pt_4plus': [],
    }

    all_perfect_efficiency = {
        'all_data': [],
        'high_pt': [],
        'high_pt_3plus': [],
        'high_pt_4plus': [],
    }

    print(f"Total dataset size: {len(dataset)}")
    print(f"Using pt thresholds: {pt_thres}")
    print(f"Using distance metric: {dist_metric}")

    def mask_func(cluster_ids, pts, pt_thres):
        if USE_ONLY_TRUE_TRACKS:
            return (cluster_ids != -1) & (pts > pt_thres)
        else:
            return (pts > pt_thres)

    for idx, evt in tqdm(enumerate(dataset), desc=f"Processing {len(dataset)} events"):
        with torch.no_grad():
            data = Batch.from_data_list([evt]).to(device)
            emb = model(data)
            
            dm_mask = mask_func(data.particle_id, data.pt, pt_thres=0.0)
            dm_results = calculate_efficiency(emb, data.particle_id, dm_mask, dist_metric, data.pt, data.x[:, 2])
            for category in all_dm_efficiency.keys():
                all_dm_efficiency[category].append(dm_results['dm_efficiency'][category])
            
            for category in all_lhc_efficiency.keys():
                all_lhc_efficiency[category].append(dm_results['lhc_efficiency'][category])
                
            for category in all_perfect_efficiency.keys():
                all_perfect_efficiency[category].append(dm_results['perfect_efficiency'][category])
            
            for pt_threshold in pt_thres:
                batch_mask = mask_func(data.particle_id, data.pt, pt_threshold)
                track_results, noise_analysis = analyze_track_performance(emb, data.particle_id, batch_mask, dist_metric, pt_threshold, idx, data.pt)
                all_track_results[pt_threshold].extend(track_results)

    for pt_threshold in pt_thres:
        if all_track_results[pt_threshold]:          
            print_track_summary(all_track_results[pt_threshold], pt_threshold)

    print_efficiency(all_dm_efficiency, all_lhc_efficiency, all_perfect_efficiency)

    output_csv = output_dir / "track_performance_results.csv"
    track_df = save_track_results_to_csv(all_track_results, output_csv)

    if VISUALS == 1:
        generate_track_visualizations(output_dir, output_csv)

    print(f"\nResults saved to: {output_dir}")

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()