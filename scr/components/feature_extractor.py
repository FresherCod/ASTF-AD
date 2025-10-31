# src/components/feature_extractor.py

import torch
from torchvision.transforms import v2 as T
from PIL import Image

# Sử dụng một dictionary toàn cục để cache các model đã được tải,
# tránh việc phải tải lại từ PyTorch Hub nhiều lần trong cùng một lần chạy script.
LOADED_MODELS = {}

def get_dinov2_features(
    image_path: str,
    model_name: str = 'dinov2_vits14',
    device: str = 'cuda'
) -> torch.Tensor | None:
    """
    Tải (nếu cần) một mô hình DINOv2 và trích xuất các feature map từ một ảnh.

    Args:
        image_path (str): Đường dẫn đến file ảnh.
        model_name (str): Tên của mô hình DINOv2 cần sử dụng.
                          'dinov2_vits14' là nhỏ và nhanh nhất.
                          'dinov2_vitb14' là cân bằng.
                          'dinov2_vitl14' là lớn và mạnh hơn.
        device (str): Thiết bị để chạy mô hình ('cuda' hoặc 'cpu').

    Returns:
        torch.Tensor | None: Một tensor chứa feature map của ảnh (shape [C, H, W]),
                             hoặc None nếu có lỗi xảy ra.
    """
    global LOADED_MODELS

    # 1. Tải mô hình DINOv2 từ PyTorch Hub (chỉ lần đầu tiên)
    if model_name not in LOADED_MODELS:
        print(f"\nLoading DINOv2 model '{model_name}' for the first time...")
        print("This may take a moment as the model is downloaded from the internet.")
        try:
            # torch.hub.load sẽ tự động tìm, tải và cache model vào ~/.cache/torch/hub
            model = torch.hub.load('facebookresearch/dinov2', model_name, verbose=False)
            model.eval().to(device)
            LOADED_MODELS[model_name] = model
            print(f"Model '{model_name}' loaded and moved to {device}")
        except Exception as e:
            print(f"Error loading model from PyTorch Hub: {e}. Please ensure you have an internet connection.")
            return None
    else:
        # Lấy model đã được tải từ cache trong bộ nhớ
        model = LOADED_MODELS[model_name]

    # 2. Định nghĩa các phép biến đổi ảnh (phải giống hệt với DINOv2)
    transform = T.Compose([
        T.ToImage(),  # Chuyển PIL Image thành Image Tensor (uint8)
        T.ToDtype(torch.float32, scale=True), # Chuyển sang float32 và scale về [0, 1]
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(224),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # 3. Tải và biến đổi ảnh
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


    # 4. Trích xuất đặc trưng
    with torch.no_grad():
        # Lấy các feature map từ lớp cuối cùng của Transformer
        features_dict = model.get_intermediate_layers(img_tensor, n=1, return_class_token=False)
        
        feature_map = features_dict[0].squeeze(0) # Bỏ batch dimension, shape: [num_patches, C]

        # Reshape lại để có dạng không gian [C, H, W]
        # Kích thước patch của DINOv2 là 14. Ảnh 224x224 -> feature map 16x16
        num_patches_side = 224 // model.patch_size
        feature_dim = feature_map.shape[-1]
        
        # Chuyển từ [num_patches, dim] -> [dim, num_patches] -> [dim, H_feat, W_feat]
        feature_map = feature_map.permute(1, 0).reshape(feature_dim, num_patches_side, num_patches_side)

    return feature_map