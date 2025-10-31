# src/02_train_adapters.py
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from rich.console import Console

# Import từ thư mục components
# Thêm đường dẫn của thư mục cha (src) vào sys.path để có thể import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from components.feature_extractor import get_dinov2_features
from components.adapter import Adapter, CompactnessLoss

# --- Cấu hình Đường dẫn ---
PROJECT_ROOT = r"D:\scr\ASTF-AD"
MVTEC_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "mvtec_ad")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ADAPTER_SAVE_DIR = os.path.join(MODELS_DIR, "adapters")

# --- Class Dataset (Không đổi) ---
class GoodImagePairDataset(Dataset):
    """Dataset tùy chỉnh để lấy ra các cặp ảnh "good" ngẫu nhiên."""
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img1_path = self.image_files[idx]
        rand_idx = torch.randint(0, len(self.image_files), (1,)).item()
        while rand_idx == idx:
            rand_idx = torch.randint(0, len(self.image_files), (1,)).item()
        img2_path = self.image_files[rand_idx]
        return img1_path, img2_path

# --- Hàm Huấn luyện (Không đổi logic, chỉ đường dẫn) ---
def train_adapter_for_category(category: str, epochs: int = 10):
    """Huấn luyện Adapter cho một category cụ thể."""
    console = Console()
    console.print(f"\n[bold yellow]---> Training Adapter for: {category.upper()} <---[/bold yellow]")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dir = os.path.join(MVTEC_DATA_PATH, category, "train", "good")

    # Kiểm tra xem có ảnh trong thư mục train không
    if not any(f.endswith(('.png', '.jpg', '.jpeg', '.bmp')) for f in os.listdir(train_dir)):
        console.print(f"[red]No training images found in {train_dir}. Skipping.[/red]")
        return

    # Chuẩn bị Data
    dataset = GoodImagePairDataset(root_dir=train_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0) # num_workers=0 là an toàn nhất

    # Khởi tạo Model, Loss, Optimizer
    adapter = Adapter().to(DEVICE)
    criterion = CompactnessLoss()
    optimizer = optim.Adam(adapter.parameters(), lr=1e-4)
    
    # Vòng lặp Huấn luyện
    adapter.train()
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for img1_paths, img2_paths in progress_bar:
            optimizer.zero_grad()

            batch_features1, batch_features2 = [], []
            for i in range(len(img1_paths)):
                feat1_map = get_dinov2_features(img1_paths[i], device=DEVICE)
                feat2_map = get_dinov2_features(img2_paths[i], device=DEVICE)
                
                if feat1_map is None or feat2_map is None: continue # Bỏ qua nếu có lỗi tải ảnh
                
                feat1_vec = feat1_map.mean(dim=[1, 2])
                feat2_vec = feat2_map.mean(dim=[1, 2])
                batch_features1.append(feat1_vec)
                batch_features2.append(feat2_vec)

            if not batch_features1: continue # Bỏ qua batch nếu không có feature nào được trích xuất

            features1 = torch.stack(batch_features1)
            features2 = torch.stack(batch_features2)
            
            adapted_features1 = adapter(features1)
            adapted_features2 = adapter(features2)
            
            loss = criterion(adapted_features1, adapted_features2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)
        console.print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
    
    # Lưu Adapter đã huấn luyện
    os.makedirs(ADAPTER_SAVE_DIR, exist_ok=True)
    adapter_save_path = os.path.join(ADAPTER_SAVE_DIR, f"{category}_adapter.pth")
    torch.save(adapter.state_dict(), adapter_save_path)
    console.print(f"[green]Adapter for '{category}' saved to {adapter_save_path}[/green]")


def main():
    """Hàm chính để lặp qua tất cả các category và huấn luyện adapter nếu cần."""
    console = Console()
    
    try:
        all_categories = sorted([d for d in os.listdir(MVTEC_DATA_PATH) if os.path.isdir(os.path.join(MVTEC_DATA_PATH, d))])
    except FileNotFoundError:
        console.print(f"[bold red]Dataset not found at {MVTEC_DATA_PATH}. Please check the path.[/bold red]")
        sys.exit(1)
        
    console.print(f"--- Starting Adapter Training for {len(all_categories)} categories ---")
    console.print(f"Trained adapters will be saved in: {ADAPTER_SAVE_DIR}")
    
    for category in all_categories:
        adapter_path = os.path.join(ADAPTER_SAVE_DIR, f"{category}_adapter.pth")
        if os.path.exists(adapter_path):
            console.print(f"Adapter for '{category}' already exists. [bold green]Skipping.[/bold green]")
            continue
        
        # Nếu chưa tồn tại, chạy huấn luyện
        train_adapter_for_category(category=category, epochs=10)
        
    console.print("\n[bold blue]All required adapters are present or have been trained.[/bold blue]")

if __name__ == '__main__':
    # freeze_support() không cần thiết vì num_workers=0
    main()