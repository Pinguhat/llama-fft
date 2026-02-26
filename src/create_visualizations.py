"""
Visualization script for FFT Block-Circulant benchmark results.
Creates publication-quality figures for the seminar presentation.

Requires: pip install matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set up nice styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.figsize'] = (10, 6)

# Data from our benchmarks
data = {
    "Original": {"layers": 0, "B": "-", "calib": "-", "cos": 1.0, "kl": 0.0, "top1": 1.0},
    
    # 1 Layer without calibration
    "1L B64 nocal": {"layers": 1, "B": 64, "calib": False, "cos": 0.890, "kl": 1.237, "top1": 0.55},
    "1L B128 nocal": {"layers": 1, "B": 128, "calib": False, "cos": 0.890, "kl": 1.236, "top1": 0.55},
    "1L B256 nocal": {"layers": 1, "B": 256, "calib": False, "cos": 0.890, "kl": 1.236, "top1": 0.55},
    
    # 1 Layer with calibration
    "1L B64 cal": {"layers": 1, "B": 64, "calib": True, "cos": 0.815, "kl": 0.326, "top1": 1.0},
    "1L B128 cal": {"layers": 1, "B": 128, "calib": True, "cos": 0.857, "kl": 0.662, "top1": 1.0},
    "1L B256 cal": {"layers": 1, "B": 256, "calib": True, "cos": 0.824, "kl": 3.103, "top1": 0.15},
    
    # 4 Layers without calibration
    "4L B64 nocal": {"layers": 4, "B": 64, "calib": False, "cos": 0.318, "kl": 5.771, "top1": 0.0},
    "4L B128 nocal": {"layers": 4, "B": 128, "calib": False, "cos": 0.318, "kl": 5.772, "top1": 0.0},
    
    # 8 Layers with calibration
    "8L B64 cal": {"layers": 8, "B": 64, "calib": True, "cos": 0.793, "kl": 3.299, "top1": 0.0},
    "8L B128 cal": {"layers": 8, "B": 128, "calib": True, "cos": 0.804, "kl": 3.125, "top1": 0.0},
}


def plot_quality_vs_layers():
    """Bar chart: Quality metrics by number of patched layers."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Group data by layer count
    configs = [
        ("Original", 0, 1.0, 0.0, 1.0),
        ("1 Layer\n(no calib)", 1, 0.890, 1.237, 0.55),
        ("1 Layer\n(calibrated B=64)", 1, 0.815, 0.326, 1.0),
        ("4 Layers\n(no calib)", 4, 0.318, 5.771, 0.0),
        ("8 Layers\n(calibrated)", 8, 0.804, 3.125, 0.0),
    ]
    
    labels = [c[0] for c in configs]
    cos_vals = [c[2] for c in configs]
    kl_vals = [c[3] for c in configs]
    top1_vals = [c[4] for c in configs]
    
    colors = ['green', 'steelblue', 'orange', 'salmon', 'salmon']
    
    # Cosine Similarity
    ax = axes[0]
    bars = ax.bar(labels, cos_vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Logit Cosine Similarity\n(higher = better)')
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Acceptable threshold')
    ax.tick_params(axis='x', rotation=30)
    
    # KL Divergence
    ax = axes[1]
    bars = ax.bar(labels, kl_vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence from Original\n(lower = better)')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Acceptable threshold')
    ax.tick_params(axis='x', rotation=30)
    
    # Top-1 Accuracy
    ax = axes[2]
    bars = ax.bar(labels, top1_vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Top-1 Accuracy')
    ax.set_title('Next-Token Prediction Match\n(higher = better)')
    ax.tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    return fig


def plot_calibration_effect():
    """Compare calibrated vs non-calibrated for 1 layer."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    block_sizes = ['B=64', 'B=128', 'B=256']
    nocal_kl = [1.237, 1.236, 1.236]
    cal_kl = [0.326, 0.662, 3.103]
    nocal_top1 = [0.55, 0.55, 0.55]
    cal_top1 = [1.0, 1.0, 0.15]
    
    x = range(len(block_sizes))
    width = 0.35
    
    # KL Divergence
    ax = axes[0]
    ax.bar([i - width/2 for i in x], nocal_kl, width, label='Without Calibration', color='steelblue')
    ax.bar([i + width/2 for i in x], cal_kl, width, label='With Calibration', color='orange')
    ax.set_ylabel('KL Divergence')
    ax.set_title('Effect of Calibration on KL Divergence\n(1 Layer patched)')
    ax.set_xticks(x)
    ax.set_xticklabels(block_sizes)
    ax.legend()
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    # Top-1 Accuracy
    ax = axes[1]
    ax.bar([i - width/2 for i in x], nocal_top1, width, label='Without Calibration', color='steelblue')
    ax.bar([i + width/2 for i in x], cal_top1, width, label='With Calibration', color='orange')
    ax.set_ylabel('Top-1 Accuracy')
    ax.set_title('Effect of Calibration on Top-1 Accuracy\n(1 Layer patched)')
    ax.set_xticks(x)
    ax.set_xticklabels(block_sizes)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    return fig


def plot_error_accumulation():
    """Show how errors accumulate with more layers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    layers = [0, 1, 4, 8]
    cos_nocal = [1.0, 0.890, 0.318, None]  # No 8L nocal data
    cos_cal = [1.0, 0.857, None, 0.804]    # No 4L cal data
    
    # Plot what we have
    ax.plot([0, 1, 4], [1.0, 0.890, 0.318], 'o-', label='Without Calibration', 
            color='steelblue', markersize=10, linewidth=2)
    ax.plot([0, 1, 8], [1.0, 0.857, 0.804], 's--', label='With Calibration', 
            color='orange', markersize=10, linewidth=2)
    
    ax.set_xlabel('Number of Patched Layers')
    ax.set_ylabel('Cosine Similarity to Original')
    ax.set_title('Error Accumulation with Layer Depth')
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Quality threshold')
    
    # Annotate
    ax.annotate('Acceptable\nquality', xy=(1, 0.85), fontsize=9, color='gray')
    ax.annotate('Catastrophic\ndegradation', xy=(4, 0.35), fontsize=9, color='red')
    
    plt.tight_layout()
    return fig


def main():
    output_dir = Path(__file__).parent / "results_summary"
    output_dir.mkdir(exist_ok=True)
    
    print("Creating visualizations...")
    
    # Plot 1: Quality by layer count
    fig1 = plot_quality_vs_layers()
    path1 = output_dir / "fig1_quality_by_layers.png"
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {path1}")
    
    # Plot 2: Calibration effect
    fig2 = plot_calibration_effect()
    path2 = output_dir / "fig2_calibration_effect.png"
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {path2}")
    
    # Plot 3: Error accumulation
    fig3 = plot_error_accumulation()
    path3 = output_dir / "fig3_error_accumulation.png"
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    print(f"Saved: {path3}")
    
    print("\nAll figures saved to results_summary/")
    print("You can include these in your seminar presentation.")
    
    # Show plots if running interactively
    plt.show()


if __name__ == "__main__":
    main()
