import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import torch
import io
from PIL import Image


def tensor_to_rgb(tensor, band_order=(0, 1, 2)):
    """
    Converts a [C, H, W] torch tensor into a numpy RGB image for display.
    band_order: which bands map to R, G, B. For LISS-IV (Green, Red, NIR):
        Natural color-like: (1, 0, 2) => Red, Green, NIR-as-blue
        False color (standard): (2, 1, 0) => NIR, Red, Green
    """
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.array(tensor)
    
    # Select bands
    r = arr[band_order[0]]
    g = arr[band_order[1]]
    b = arr[band_order[2]]
    
    rgb = np.stack([r, g, b], axis=-1)  # [H, W, 3]
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def tensor_to_single_band(tensor, band=0, cmap='gray'):
    """Converts a single band from [C, H, W] tensor to a colorized numpy image."""
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.array(tensor)
    
    if arr.ndim == 3:
        band_data = arr[band]
    else:
        band_data = arr
    
    return band_data


def create_comparison_figure(results, figsize=(20, 10)):
    """
    Creates a multi-panel matplotlib comparison figure showing:
    Row 1: Cloudy Input | Cloud Mask | TPT Prediction | SAR Fused
    Row 2: Reconstructed | Ground Truth | Difference | Uncertainty
    
    Returns a PIL Image.
    """
    fig, axes = plt.subplots(2, 4, figsize=figsize, facecolor='#1a1a2e')
    
    # Row 1
    # Cloudy input (false color: NIR-Red-Green for visualization)
    cloudy_rgb = tensor_to_rgb(results['inputs']['cloudy'], band_order=(2, 1, 0))
    axes[0, 0].imshow(cloudy_rgb)
    axes[0, 0].set_title('Cloudy Input (False Color)', color='white', fontsize=11, fontweight='bold')
    
    # Cloud/Shadow mask
    mask = results['intermediates']['detected_mask']
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
    axes[0, 1].imshow(mask, cmap='Reds', vmin=0, vmax=1)
    axes[0, 1].set_title('Detected Cloud+Shadow Mask', color='white', fontsize=11, fontweight='bold')
    
    # TPT prediction
    tpt_rgb = tensor_to_rgb(results['intermediates']['pred_tpt'], band_order=(2, 1, 0))
    axes[0, 2].imshow(tpt_rgb)
    axes[0, 2].set_title('TPT Prediction', color='white', fontsize=11, fontweight='bold')
    
    # SAR fused
    sar_rgb = tensor_to_rgb(results['intermediates']['sar_fused'], band_order=(2, 1, 0))
    axes[0, 3].imshow(sar_rgb)
    axes[0, 3].set_title('SAR Coherence Fused', color='white', fontsize=11, fontweight='bold')
    
    # Row 2
    # Reconstructed
    recon_rgb = tensor_to_rgb(results['outputs']['final_output'], band_order=(2, 1, 0))
    axes[1, 0].imshow(recon_rgb)
    axes[1, 0].set_title('Reconstructed Output', color='#00ff88', fontsize=11, fontweight='bold')
    
    # Ground truth
    gt_rgb = tensor_to_rgb(results['outputs']['gt'], band_order=(2, 1, 0))
    axes[1, 1].imshow(gt_rgb)
    axes[1, 1].set_title('Ground Truth', color='#00ff88', fontsize=11, fontweight='bold')
    
    # Absolute difference map
    diff = np.abs(recon_rgb - gt_rgb)
    diff_mean = np.mean(diff, axis=-1)
    axes[1, 2].imshow(diff_mean, cmap='hot', vmin=0, vmax=0.5)
    axes[1, 2].set_title('|Reconstruction - GT|', color='white', fontsize=11, fontweight='bold')
    
    # Uncertainty map
    uncertainty = results['intermediates']['uncertainty']
    if isinstance(uncertainty, torch.Tensor):
        uncertainty = uncertainty.detach().cpu().numpy()
    if uncertainty.ndim == 3:
        uncertainty = uncertainty[0]
    axes[1, 3].imshow(uncertainty, cmap='viridis', vmin=0)
    axes[1, 3].set_title('Uncertainty Map', color='white', fontsize=11, fontweight='bold')
    
    # Style all axes
    for ax in axes.flat:
        ax.axis('off')
        ax.set_facecolor('#1a1a2e')
    
    plt.tight_layout(pad=1.5)
    
    # Convert to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    pil_img = Image.open(buf).copy()
    plt.close(fig)
    buf.close()
    
    return pil_img


def create_metrics_bar_chart(metrics, figsize=(8, 5)):
    """Creates a bar chart for PSNR, SSIM, SAM values."""
    fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')
    
    names = ['PSNR (dB)', 'SSIM', 'SAM (rad)']
    values = [metrics['psnr'], metrics['ssim'], metrics['sam']]
    colors = ['#0ea5e9', '#22c55e', '#f59e0b']
    
    bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=0.5, width=0.5)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold', color='white')
    
    ax.set_ylabel('Value', color='white', fontsize=12)
    ax.set_title('Reconstruction Quality Metrics', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    pil_img = Image.open(buf).copy()
    plt.close(fig)
    buf.close()
    
    return pil_img


def create_ndvi_map(tensor, figsize=(6, 5)):
    """
    Creates an NDVI visualization from a LISS-IV [3, H, W] tensor.
    Band 0 = Green, Band 1 = Red, Band 2 = NIR
    NDVI = (NIR - Red) / (NIR + Red + eps)
    """
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.array(tensor)
    
    nir = arr[2]
    red = arr[1]
    ndvi = (nir - red) / (nir + red + 1e-8)
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a2e')
    im = ax.imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.9)
    ax.set_title('NDVI Map', color='white', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    pil_img = Image.open(buf).copy()
    plt.close(fig)
    buf.close()
    
    return pil_img
