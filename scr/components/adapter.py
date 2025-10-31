# src/components/adapter.py

import torch
import torch.nn as nn

class Adapter(nn.Module):
    """
    Một Adapter Module đơn giản để tinh chỉnh các đặc trưng từ một Foundation Model.

    Kiến trúc này sử dụng một bottleneck (nút cổ chai) và một kết nối phần dư
    (residual connection) để học một cách hiệu quả và ổn định.
    """
    def __init__(self, input_dim: int = 384, hidden_dim: int = 128):
        """
        Khởi tạo các lớp của Adapter.

        Args:
            input_dim (int): Số chiều của vector đặc trưng đầu vào (từ DINOv2). Mặc định là 384.
            hidden_dim (int): Số chiều của lớp ẩn (bottleneck). Mặc định là 128.
        """
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện quá trình forward pass.

        Args:
            features (torch.Tensor): Tensor đặc trưng đầu vào, có shape [N, C],
                                     trong đó N là số patch (hoặc batch size),
                                     C là số chiều đặc trưng (input_dim).
        
        Returns:
            torch.Tensor: Tensor đặc trưng đã được tinh chỉnh, có cùng shape với đầu vào.
        """
        # Giữ lại giá trị gốc để tạo kết nối phần dư
        residual = features
        
        # Cho đi qua các lớp của Adapter
        x = self.layer1(features)
        x = self.relu(x)
        x = self.layer2(x)
        
        # Cộng lại với giá trị gốc
        output = x + residual
        return output

class CompactnessLoss(nn.Module):
    """
    Hàm loss dựa trên Cosine Similarity để "gom cụm" (compact) các đặc trưng "good".

    Mục tiêu là tối đa hóa sự tương đồng (cosine similarity) giữa các vector
    đặc trưng của các mẫu ảnh "good" khác nhau, từ đó làm cho không gian
    đặc trưng của các mẫu "good" trở nên cô đặc hơn.
    """
    def __init__(self):
        super().__init__()
        # Khởi tạo hàm tính Cosine Similarity trên chiều đặc trưng (dimension 1)
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Tính toán loss.

        Args:
            features1 (torch.Tensor): Batch các đặc trưng từ nhóm ảnh 1. Shape [B, C].
            features2 (torch.Tensor): Batch các đặc trưng từ nhóm ảnh 2. Shape [B, C].

        Returns:
            torch.Tensor: Giá trị loss (một số vô hướng).
        """
        # Tính toán cosine similarity giữa các cặp vector tương ứng trong batch
        similarity = self.cosine_similarity(features1, features2)
        
        # Loss được định nghĩa là 1 trừ đi giá trị similarity trung bình.
        # Tối thiểu hóa loss này tương đương với việc tối đa hóa similarity (đẩy về 1).
        loss = 1 - similarity.mean()
        return loss