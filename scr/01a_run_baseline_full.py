# src/01a_run_baseline_full.py
import os, sys, torch, shutil
from multiprocessing import freeze_support
from rich.console import Console
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from torchvision.transforms import v2 as T

# --- Cấu hình Đường dẫn ---
PROJECT_ROOT = r"D:\scr\ASTF-AD"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models") # Thư mục model mới
MVTEC_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "mvtec_ad")

def run_patchcore_for_category(category: str, console: Console):
    """
    Huấn luyện và kiểm tra PatchCore, sau đó di chuyển checkpoint vào thư mục models/.
    """
    console.print(f"\n[bold cyan]Running Full Baseline for: {category.upper()}[/bold cyan]")
    try:
        model = Patchcore(layers=["layer2", "layer3"], backbone="wide_resnet50_2")
        augmentations = T.Compose([
            T.Resize((256, 256), antialias=True), T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        datamodule = MVTecAD(
            root=MVTEC_DATA_PATH, category=category, train_batch_size=8, eval_batch_size=8, num_workers=4,
            train_augmentations=augmentations, val_augmentations=augmentations, test_augmentations=augmentations
        )
        
        # Tạm thời lưu kết quả vào một thư mục tạm
        temp_result_path = os.path.join(RESULTS_DIR, "temp", category)

        engine = Engine(
            accelerator="auto", devices=1, default_root_dir=temp_result_path,
            logger=True, enable_checkpointing=True, max_epochs=1
        )
        
        console.print(f"Fitting model...")
        engine.fit(model=model, datamodule=datamodule)
        
        console.print(f"Testing model...")
        engine.test(model=model, datamodule=datamodule)
        
        # --- Di chuyển Checkpoint vào thư mục models/ ---
        checkpoint_path = engine.trainer.checkpoint_callback.best_model_path
        if not checkpoint_path: # Nếu không có best_model, lấy last.ckpt
             checkpoint_path = os.path.join(temp_result_path, "weights", "last.ckpt")

        if os.path.exists(checkpoint_path):
            target_dir = os.path.join(MODELS_DIR, "patchcore")
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, f"{category}.ckpt")
            shutil.copy(checkpoint_path, target_path)
            console.print(f"[green]Checkpoint saved to {target_path}[/green]")
        
        # Xóa thư mục tạm
        shutil.rmtree(temp_result_path)

    except Exception as e:
        console.print(f"[bold red]An error occurred while processing {category}: {e}[/bold red]")

def main():
    console = Console()
    try:
        all_categories = sorted([d for d in os.listdir(MVTEC_DATA_PATH) if os.path.isdir(os.path.join(MVTEC_DATA_PATH, d))])
    except FileNotFoundError:
        console.print(f"[bold red]Dataset not found at {MVTEC_DATA_PATH}.[/bold red]"); sys.exit(1)
        
    console.print("--- Starting Full PatchCore Baseline Benchmark ---")
    for category in all_categories:
        run_patchcore_for_category(category, console)
    
    console.print("\n[bold yellow]Full benchmark run completed. Checkpoints are saved in 'models/patchcore/'.[/bold yellow]")
    console.print("[bold yellow]Now, run '01b_summarize_baseline.py' to generate the summary CSV.[/bold yellow]")

if __name__ == '__main__':
    freeze_support()
    main()