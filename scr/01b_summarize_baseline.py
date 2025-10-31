'''*   **Mục tiêu:** Chỉ chạy phần đánh giá (test) dựa trên các checkpoint đã có trong `models/patchcore/`, sau đó tạo file tổng hợp.
'''
# src/01b_summarize_baseline.py
import os, sys, torch, pandas as pd, numpy as np
from multiprocessing import freeze_support
from rich.console import Console
from rich.table import Table
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from torchvision.transforms import v2 as T

# --- Cấu hình Đường dẫn ---
PROJECT_ROOT = r"D:\scr\ASTF-AD"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MVTEC_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "mvtec_ad")

def test_and_get_metrics(category: str, console: Console):
    """
    Tải checkpoint của PatchCore, chạy test, và trả về kết quả.
    """
    console.print(f"\n[bold cyan]Testing Baseline for: {category.upper()}[/bold cyan]")
    
    checkpoint_path = os.path.join(MODELS_DIR, "patchcore", f"{category}.ckpt")
    if not os.path.exists(checkpoint_path):
        console.print(f"[bold red]Checkpoint for {category} not found at {checkpoint_path}. Skipping.[/bold red]")
        return None
        
    try:
        model = Patchcore.load_from_checkpoint(checkpoint_path)
        augmentations = T.Compose([
            T.Resize((256, 256), antialias=True), T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        datamodule = MVTecAD(
            root=MVTEC_DATA_PATH, category=category, train_batch_size=8, eval_batch_size=8, num_workers=4,
            train_augmentations=augmentations, val_augmentations=augmentations, test_augmentations=augmentations
        )
        engine = Engine(accelerator="auto", devices=1, logger=False) # Tắt logger vì chỉ test
        
        test_results = engine.test(model=model, datamodule=datamodule)
        
        if test_results:
            result_dict = test_results[0]
            result_dict['category'] = category
            return result_dict
            
    except Exception as e:
        console.print(f"[bold red]An error occurred while testing {category}: {e}[/bold red]")
        return None

def main():
    console = Console()
    try:
        all_categories = sorted([d for d in os.listdir(MVTEC_DATA_PATH) if os.path.isdir(os.path.join(MVTEC_DATA_PATH, d))])
    except FileNotFoundError:
        console.print(f"[bold red]Dataset not found at {MVTEC_DATA_PATH}.[/bold red]"); sys.exit(1)
    
    all_results = []
    console.print("--- Summarizing PatchCore Baseline from Checkpoints ---")
    for category in all_categories:
        result = test_and_get_metrics(category, console)
        if result:
            all_results.append(result)
            
    if not all_results:
        console.print("[bold red]No results were generated. Make sure checkpoints exist.[/bold red]"); return
        
    results_df = pd.DataFrame(all_results)
    results_df = results_df[['category', 'image_AUROC', 'pixel_AUROC']].round(4)
    numeric_cols = results_df.select_dtypes(include=np.number).columns
    average_row = results_df[numeric_cols].mean().to_dict()
    average_row['category'] = 'Average'
    results_df = pd.concat([results_df, pd.DataFrame([average_row])], ignore_index=True).round(4)
    
    table = Table(title="PatchCore Baseline (Re-tested from Checkpoints)")
    for col in results_df.columns: table.add_column(col, justify="center")
    for _, row in results_df.iterrows(): table.add_row(*[str(val) for val in row.values])
    console.print(table)
    
    summary_path = os.path.join(RESULTS_DIR, "patchcore_baseline_summary.csv")
    results_df.to_csv(summary_path, index=False)
    console.print(f"\n[bold green]Baseline summary saved to: {summary_path}[/bold green]")
    
if __name__ == '__main__':
    freeze_support()
    main()