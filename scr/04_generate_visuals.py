# src/04_generate_visuals.py
import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import v2 as T
import glob
from rich.console import Console

# Thêm đường dẫn src vào sys.path để import từ components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from components.feature_extractor import get_dinov2_features
from components.adapter import Adapter
from anomalib.models import Patchcore

# Tắt các cảnh báo không cần thiết từ matplotlib
import warnings
warnings.filterwarnings("ignore")

# --- Cấu hình Đường dẫn Toàn cục ---
PROJECT_ROOT = r"D:\scr\ASTF-AD"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MVTEC_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "mvtec_ad")
VISUALS_SAVE_DIR = os.path.join(RESULTS_DIR, "visuals")

# --- Các hàm Tiện ích (tái sử dụng từ các script trước) ---
def normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    min_val, max_val = anomaly_map.min(), anomaly_map.max()
    return (anomaly_map - min_val) / (max_val - min_val) if max_val > min_val else anomaly_map

patchcore_transform = T.Compose([
    T.ToImage(), T.Resize((256, 256), antialias=True), T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# --- BẢN ĐỒ CHIẾN LƯỢC (cần để biết phương pháp Voted) ---
STRATEGY_MAP = {
    'carpet': 'semantic_adapter', 'grid': 'semantic_adapter', 'leather': 'semantic_adapter',
    'wood': 'semantic_adapter', 'hazelnut': 'semantic_adapter',
    'metal_nut': 'fusion_weighted_sum_0.7', 'screw': 'fusion_weighted_sum_0.7',
    'tile': 'fusion_weighted_sum_0.7', 'toothbrush': 'fusion_weighted_sum_0.7',
    'transistor': 'fusion_weighted_sum_0.7',
    'bottle': 'fusion_add', 'cable': 'fusion_add', 'capsule': 'fusion_add',
    'pill': 'fusion_add', 'zipper': 'fusion_add',
}

def generate_visual_comparison(category: str, image_filename: str):
    """
    Tạo và lưu một hình ảnh so sánh trực quan cho một ảnh lỗi cụ thể.
    """
    console = Console()
    console.print(f"\n[bold blue]Generating visuals for: {category} - {image_filename}[/bold blue]")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 1. Tải các model cần thiết ---
    try:
        patchcore_checkpoint_path = os.path.join(MODELS_DIR, "patchcore", f"{category}.ckpt")
        patchcore_model = Patchcore.load_from_checkpoint(patchcore_checkpoint_path).to(DEVICE).eval()

        adapter_path = os.path.join(MODELS_DIR, "adapters", f"{category}_adapter.pth")
        adapter = Adapter().to(DEVICE).eval()
        adapter.load_state_dict(torch.load(adapter_path))
    except FileNotFoundError as e:
        console.print(f"[red]Error loading models for {category}: {e}. Skipping.[/red]"); return

    # Xây dựng semantic memory bank
    train_dir = os.path.join(MVTEC_DATA_PATH, category, "train", "good")
    semantic_memory_bank = []
    with torch.no_grad():
        for file_path in os.listdir(train_dir):
            feature_map = get_dinov2_features(os.path.join(train_dir, file_path), device=DEVICE)
            if feature_map is None: continue
            patches = feature_map.view(feature_map.shape[0], -1).T
            semantic_memory_bank.append(adapter(patches))
    semantic_memory_bank = torch.cat(semantic_memory_bank, dim=0)
    
    # --- 2. Tìm đường dẫn ảnh và mask ---
    test_dir = os.path.join(MVTEC_DATA_PATH, category, "test")
    image_path = None
    for root, _, files in os.walk(test_dir):
        if image_filename in files and 'good' not in root:
            image_path = os.path.join(root, image_filename); break
    if not image_path:
        console.print(f"[red]Image '{image_filename}' not found in test set for '{category}'.[/red]"); return

    mask_path = image_path.replace('/test/', '/ground_truth/').replace('\\test\\', '\\ground_truth\\').replace('.png', '_mask.png')
    if not os.path.exists(mask_path): mask_path = mask_path.replace('_mask.png', '.png')
    
    # --- 3. Tính toán các Anomaly Map ---
    with torch.no_grad():
        pil_image = Image.open(image_path).convert("RGB")
        target_size = pil_image.size[::-1]

        # Map 1: PatchCore (Baseline)
        image_tensor = patchcore_transform(pil_image).unsqueeze(0).to(DEVICE)
        map_patchcore = patchcore_model(image_tensor)[2].squeeze().cpu().numpy()

        # Map 2: Semantic + Adapter
        feature_map = get_dinov2_features(image_path, device=DEVICE)
        patches = feature_map.view(feature_map.shape[0], -1).T
        adapted_patches = adapter(patches)
        distances = torch.cdist(adapted_patches, semantic_memory_bank)
        map_semantic = torch.min(distances, dim=1)[0].view(feature_map.shape[1], feature_map.shape[2]).cpu().numpy()

        # Map 3: Voted Strategy
        strategy = STRATEGY_MAP.get(category, 'fusion_add')
        if strategy == 'semantic_adapter': final_map_unresized = map_semantic
        elif strategy == 'patchcore': final_map_unresized = map_patchcore
        else:
            norm_pc = normalize_map(map_patchcore)
            resized_sem = T.functional.resize(torch.tensor(map_semantic).unsqueeze(0), size=map_patchcore.shape, interpolation=T.InterpolationMode.BILINEAR, antialias=True).squeeze(0).numpy()
            norm_sem = normalize_map(resized_sem)
            if strategy == 'fusion_add': final_map_unresized = norm_pc + norm_sem
            elif strategy == 'fusion_weighted_sum_0.7': final_map_unresized = (0.7 * norm_pc) + (0.3 * norm_sem)
        
    # --- 4. Vẽ và lưu hình ảnh ---
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f"Anomaly Map Comparison: {category} - {image_filename}", fontsize=16)
    
    maps_to_plot = {
        "PatchCore (Baseline)": map_patchcore,
        "Semantic + Adapter": map_semantic,
        f"ASTF-AD System ({strategy})": final_map_unresized
    }
    
    # Panel 1: Original Image
    axes[0].imshow(pil_image); axes[0].set_title("Original Image"); axes[0].axis('off')

    # Panel 2: Ground Truth
    if os.path.exists(mask_path): axes[1].imshow(Image.open(mask_path).convert("L"), cmap='gray')
    axes[1].set_title("Ground Truth"); axes[1].axis('off')

    # Panels 3-5: Anomaly Maps
    i = 2
    for name, map_data in maps_to_plot.items():
        resized_map = T.functional.resize(torch.tensor(map_data).unsqueeze(0), size=list(target_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True).squeeze(0).numpy()
        im = axes[i].imshow(resized_map, cmap='jet')
        axes[i].set_title(name); axes[i].axis('off')
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        i += 1
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(VISUALS_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(VISUALS_SAVE_DIR, f"{category}_{image_filename.replace('.png', '')}_comparison.png")
    plt.savefig(save_path)
    console.print(f"[green]Visual comparison saved to {save_path}[/green]")
    plt.close(fig)

if __name__ == '__main__':
    # --- DANH SÁCH CÁC ẢNH CẦN TRỰC QUAN HÓA ---
    # Bạn hãy tự chọn ra những file ảnh lỗi tiêu biểu nhất.
    # Key: Tên category, Value: Tên file ảnh (bao gồm đuôi .png)
    images_to_visualize = {
        'carpet': '002.png',      # Category mà Semantic+Adapter vượt trội
        'screw': '001.png',       # Category mà PatchCore rất mạnh
        'bottle': '001.png',      # Category mà Fusion được áp dụng
        'transistor': '001.png' # Category mà hệ thống hoạt động chưa tốt
    }

    console = Console()
    for category, filename in images_to_visualize.items():
        # Kiểm tra xem các model cần thiết có tồn tại không trước khi chạy
        patchcore_ckpt = os.path.join(MODELS_DIR, "patchcore", f"{category}.ckpt")
        adapter_pth = os.path.join(MODELS_DIR, "adapters", f"{category}_adapter.pth")
        if not os.path.exists(patchcore_ckpt) or not os.path.exists(adapter_pth):
            console.print(f"[yellow]Models for '{category}' not found. Skipping visual generation.[/yellow]")
            continue
        generate_visual_comparison(category, filename)