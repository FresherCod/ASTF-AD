# src/03_evaluate_system.py
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import v2 as T
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
import glob
import pandas as pd
from rich.console import Console
from rich.table import Table
from multiprocessing import freeze_support

# Thêm đường dẫn src vào sys.path để import từ components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from components.feature_extractor import get_dinov2_features
from components.adapter import Adapter
from anomalib.models import Patchcore


# --- Cấu hình Đường dẫn Toàn cục ---
PROJECT_ROOT = r"D:\scr\ASTF-AD"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MVTEC_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "mvtec_ad")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Các hàm Tiện ích ---
def normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    min_val, max_val = anomaly_map.min(), anomaly_map.max()
    return (anomaly_map - min_val) / (max_val - min_val) if max_val > min_val else anomaly_map

patchcore_transform = T.Compose([
    T.ToImage(), T.Resize((256, 256), antialias=True), T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# --- BẢN ĐỒ CHIẾN LƯỢC (STRATEGY MAP) ---
STRATEGY_MAP = {
    'carpet':   'semantic_adapter', 'grid': 'semantic_adapter',
    'leather':  'semantic_adapter', 'wood': 'semantic_adapter',
    'hazelnut': 'semantic_adapter',
    'metal_nut':'fusion_weighted_sum_0.7', 'screw': 'fusion_weighted_sum_0.7',
    'tile':     'fusion_weighted_sum_0.7', 'toothbrush': 'fusion_weighted_sum_0.7',
    'transistor':'fusion_weighted_sum_0.7',
    'bottle':   'fusion_add', 'cable': 'fusion_add',
    'capsule':  'fusion_add', 'pill': 'fusion_add',
    'zipper':   'fusion_add',
}

def run_expert_selection_test(category: str, patchcore_model, adapter, semantic_memory_bank):
    """
    Chạy đánh giá cho một category, sử dụng các model đã được tải sẵn.
    """
    strategy = STRATEGY_MAP.get(category, 'fusion_add')
    
    test_dir = os.path.join(MVTEC_DATA_PATH, category, "test")
    test_files, ground_truth_masks, image_labels = [], [], []
    for defect_type in os.listdir(test_dir):
        defect_dir = os.path.join(test_dir, defect_type)
        if not os.path.isdir(defect_dir): continue
        for file_name in os.listdir(defect_dir):
            if not file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp')): continue
            image_path = os.path.join(defect_dir, file_name)
            test_files.append(image_path)
            is_good = (defect_type == 'good'); image_labels.append(0 if is_good else 1)
            with Image.open(image_path) as img: img_size = img.size[::-1]
            if is_good: ground_truth_masks.append(np.zeros(img_size, dtype=np.uint8))
            else:
                mask_path_1 = image_path.replace('/test/', '/ground_truth/').replace('\\test\\', '\\ground_truth\\').replace('.png', '_mask.png')
                mask_path_2 = image_path.replace('.png', '.png') # fallback for some datasets
                if os.path.exists(mask_path_1): mask_path = mask_path_1
                elif os.path.exists(mask_path_2): mask_path = mask_path_2
                else:
                    ground_truth_masks.append(np.zeros(img_size, dtype=np.uint8)); continue
                mask = np.array(Image.open(mask_path).convert('L'))
                if mask.shape != img_size: mask = np.array(Image.open(mask_path).convert('L').resize(img_size[::-1]))
                ground_truth_masks.append((mask > 0).astype(np.uint8))

    image_scores_final, pixel_scores_final = [], []
    with torch.no_grad():
        for i, file_path in enumerate(tqdm(test_files, desc=f"Applying Strategy ({strategy})")):
            pil_image = Image.open(file_path).convert("RGB"); target_size = ground_truth_masks[i].shape
            
            # --- Lấy Anomaly Map từ các nhánh ---
            anomaly_map_semantic_raw = None
            if 'semantic' in strategy or 'fusion' in strategy:
                feature_map_sem = get_dinov2_features(file_path, device=DEVICE)
                patches_sem = feature_map_sem.view(feature_map_sem.shape[0], -1).T
                adapted_patches = adapter(patches_sem)
                distances = torch.cdist(adapted_patches, semantic_memory_bank)
                min_distances, _ = torch.min(distances, dim=1)
                anomaly_map_semantic_raw = min_distances.view(feature_map_sem.shape[1], feature_map_sem.shape[2]).cpu().numpy()

            anomaly_map_patchcore = None
            if 'patchcore' in strategy or 'fusion' in strategy:
                image_tensor = patchcore_transform(pil_image).unsqueeze(0).to(DEVICE)
                patchcore_result = patchcore_model(image_tensor)
                anomaly_map_patchcore = patchcore_result[2].squeeze().cpu().numpy()
            
            # --- Áp dụng Logic của Chiến lược ---
            if strategy == 'semantic_adapter':
                final_map_unresized = anomaly_map_semantic_raw
            elif strategy == 'patchcore':
                final_map_unresized = anomaly_map_patchcore
            else: # Các chiến lược Fusion
                norm_pc = normalize_map(anomaly_map_patchcore)
                h_pc, w_pc = anomaly_map_patchcore.shape
                resized_sem = T.functional.resize(torch.tensor(anomaly_map_semantic_raw).unsqueeze(0), size=[h_pc, w_pc], interpolation=T.InterpolationMode.BILINEAR, antialias=True).squeeze(0).numpy()
                norm_sem = normalize_map(resized_sem)
                if strategy == 'fusion_add': final_map_unresized = norm_pc + norm_sem
                elif strategy == 'fusion_weighted_sum_0.7':
                    alpha = 0.7; final_map_unresized = (alpha * norm_pc) + ((1 - alpha) * norm_sem)
            
            # Xử lý hậu kỳ
            final_map_resized = T.functional.resize(torch.tensor(final_map_unresized).unsqueeze(0), size=list(target_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True).squeeze(0).numpy()
            final_map_smoothed = gaussian_filter(final_map_resized, sigma=4)
            image_scores_final.append(np.max(final_map_smoothed)); pixel_scores_final.append(final_map_smoothed.flatten())

    image_auroc = roc_auc_score(image_labels, image_scores_final)
    gt_flat = np.concatenate([m.flatten() for m in ground_truth_masks])
    scores_flat = np.concatenate(pixel_scores_final)
    pixel_auroc = roc_auc_score(gt_flat, scores_flat)
    return { "category": category, "applied_strategy": strategy, "image_AUROC": image_auroc, "pixel_AUROC": pixel_auroc }

def main():
    """Hàm chính để chạy toàn bộ benchmark."""
    console = Console()
    try:
        all_categories = sorted([d for d in os.listdir(MVTEC_DATA_PATH) if os.path.isdir(os.path.join(MVTEC_DATA_PATH, d))])
    except FileNotFoundError:
        console.print(f"[bold red]Dataset not found at {MVTEC_DATA_PATH}.[/bold red]"); sys.exit(1)
        
    final_results = []

    for category in all_categories:
        console.print(f"\n[bold magenta]Processing Category: {category.upper()}[/bold magenta]")

        # 1. Kiểm tra và Tải các model cần thiết
        patchcore_checkpoint_path = os.path.join(MODELS_DIR, "patchcore", f"{category}.ckpt")
        adapter_path = os.path.join(MODELS_DIR, "adapters", f"{category}_adapter.pth")

        if not os.path.exists(patchcore_checkpoint_path) or not os.path.exists(adapter_path):
            console.print(f"[bold red]Pre-trained models for {category} not found in 'models/' directory. Skipping.[/bold red]")
            console.print(f"Please run '01a_run_baseline_full.py' and '02_train_adapters.py' first.")
            continue
        
        patchcore_model = Patchcore.load_from_checkpoint(patchcore_checkpoint_path).to(DEVICE).eval()
        adapter = Adapter().to(DEVICE).eval()
        adapter.load_state_dict(torch.load(adapter_path))

        # 2. Xây dựng memory bank
        train_dir = os.path.join(MVTEC_DATA_PATH, category, "train", "good")
        semantic_memory_bank = []
        with torch.no_grad():
            for file_path in tqdm(os.listdir(train_dir), desc=f"Building semantic bank for {category}"):
                feature_map = get_dinov2_features(os.path.join(train_dir, file_path), device=DEVICE)
                if feature_map is None: continue
                patches = feature_map.view(feature_map.shape[0], -1).T
                semantic_memory_bank.append(adapter(patches))
        semantic_memory_bank = torch.cat(semantic_memory_bank, dim=0)

        # 3. Chạy đánh giá
        result = run_expert_selection_test(category, patchcore_model, adapter, semantic_memory_bank)
        if result: final_results.append(result)

    # 4. Tổng hợp và so sánh với baseline
    if final_results:
        voted_df = pd.DataFrame(final_results).set_index('category')
        
        baseline_path = os.path.join(RESULTS_DIR, "patchcore_baseline_summary.csv")
        if not os.path.exists(baseline_path):
            console.print(f"[bold red]Baseline file '{os.path.basename(baseline_path)}' not found. Cannot compare.[/bold red]")
            # In kết quả của hệ thống mới
            print(voted_df.to_markdown())
            return

        baseline_df = pd.read_csv(baseline_path).rename(columns={'image_AUROC': 'baseline_image_AUROC', 'pixel_AUROC': 'baseline_pixel_AUROC'}).set_index('category')
        
        comparison_df = voted_df.join(baseline_df[['baseline_image_AUROC', 'baseline_pixel_AUROC']]).round(4)
        comparison_df.loc['Average'] = comparison_df.mean(numeric_only=True).round(4)

        console.print("\n\n--- [bold green]OVERALL PERFORMANCE COMPARISON[/bold green] ---")
        try:
            import tabulate
            print(comparison_df.to_markdown())
        except ImportError:
            print(comparison_df)

        summary_path = os.path.join(RESULTS_DIR, "final_benchmark_comparison.csv")
        comparison_df.reset_index().to_csv(summary_path, index=False)
        console.print(f"\n[bold green]Final comparison summary saved to: {summary_path}[/bold green]")

if __name__ == '__main__':
    freeze_support()
    main()
'''
**Những thay đổi và cải tiến chính:**

1.  **Cấu trúc lại Đường dẫn:** Tất cả các đường dẫn đều được cập nhật để trỏ đến `models/patchcore/`, `models/adapters/`, `data/mvtec_ad/`...
2.  **Import theo Chuẩn:** Các lệnh `import` được sửa lại để lấy các class từ package `components`.
3.  **Tối ưu hóa:** Thay vì tải lại model cho mỗi lần test, script `main` bây giờ sẽ tải các model cần thiết cho mỗi category một lần duy nhất ở vòng lặp ngoài, sau đó truyền chúng vào hàm `run_expert_selection_test`. Điều này giúp tiết kiệm đáng kể thời gian tải model.
4.  **Kiểm tra File Tồn tại:** Script sẽ kiểm tra xem cả checkpoint PatchCore và Adapter có tồn tại hay không trước khi chạy. Nếu không, nó sẽ bỏ qua category đó và đưa ra thông báo hướng dẫn rõ ràng.
5.  **Logic Chạy Tường minh:** Hàm `run_expert_selection_test` bây giờ chỉ tập trung vào việc áp dụng chiến lược và tính toán, không còn phải tải model nữa.

**Cách sử dụng:**
1.  Đảm bảo bạn đã chạy `01...` và `02...` để có đầy đủ các file trọng số trong thư mục `models/`.
2.  Chạy script:
    ```bash
    python src/03_evaluate_system.py
    

Script sẽ thực hiện lại toàn bộ benchmark cho hệ thống của bạn, sử dụng các model đã được huấn luyện, và tạo ra file so sánh cuối cùng `final_benchmark_comparison.csv` trong thư mục `results/`.
'''