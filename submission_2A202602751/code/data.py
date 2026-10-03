"""data.py — Chuẩn bị và tiền xử lý dữ liệu.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    train_path = Path(processed_dir) / "train.npz"
    eval_path = Path(processed_dir) / "eval.npz"

    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file npz tại '{processed_dir}'. Vui lòng chạy scripts/split_data.py trước."
        )

    with np.load(train_path) as tr:
        X_train_full = tr["X"].astype(np.float32)
        y_train_full = tr["y"].astype(np.int64)

    with np.load(eval_path) as ev:
        X_eval = ev["X"].astype(np.float32)
        y_eval = ev["y"].astype(np.int64)
        eval_row_id = ev["row_id"].astype(np.int64)

    assert X_train_full.ndim == 2 and X_train_full.shape[1] == 54, f"X_train_full sai shape: {X_train_full.shape}"
    assert X_train_full.dtype == np.float32, f"X_train_full sai dtype: {X_train_full.dtype}"
    assert y_train_full.ndim == 1 and len(y_train_full) == len(X_train_full), f"y_train_full sai shape: {y_train_full.shape}"
    assert y_train_full.dtype == np.int64, f"y_train_full sai dtype: {y_train_full.dtype}"

    assert X_eval.ndim == 2 and X_eval.shape[1] == 54, f"X_eval sai shape: {X_eval.shape}"
    assert X_eval.dtype == np.float32, f"X_eval sai dtype: {X_eval.dtype}"
    assert y_eval.ndim == 1 and len(y_eval) == len(X_eval), f"y_eval sai shape: {y_eval.shape}"
    assert y_eval.dtype == np.int64, f"y_eval sai dtype: {y_eval.dtype}"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=seed, stratify=y
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    mean = X_tr[:, :N_NUMERIC].mean(axis=0)
    std = X_tr[:, :N_NUMERIC].std(axis=0)
    # Tránh chia cho 0 nếu std = 0
    std = np.where(std == 0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    if isinstance(X, torch.Tensor):
        X_new = X.clone()
        mean_t = torch.as_tensor(mean, device=X.device, dtype=X.dtype)
        std_t = torch.as_tensor(std, device=X.device, dtype=X.dtype)
        std_t = torch.where(std_t == 0, torch.ones_like(std_t), std_t)
        X_new[:, :N_NUMERIC] = (X_new[:, :N_NUMERIC] - mean_t) / std_t
        return X_new
    else:
        X_new = np.array(X, copy=True, dtype=np.float32)
        safe_std = np.where(std == 0, 1.0, std)
        X_new[:, :N_NUMERIC] = (X_new[:, :N_NUMERIC] - mean) / safe_std
        return X_new


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    # 1. Đọc dữ liệu
    X_train_full, y_train_full, X_eval_raw, y_eval_raw, eval_row_id = load_split(processed_dir=processed_dir)

    # Tách validation từ train (phân tầng theo nhãn)
    X_tr_raw, y_tr, X_val_raw, y_val = make_val_split(
        X_train_full, y_train_full, val_fraction=val_fraction, seed=seed
    )

    # Tính mean và std chỉ trên X_tr (chống data leakage)
    mean, std = fit_standardizer(X_tr_raw)

    # 2. Áp dụng chuẩn hoá cho cả 3 tập bằng mean/std của X_tr
    X_tr_norm = apply_standardizer(X_tr_raw, mean, std)
    X_val_norm = apply_standardizer(X_val_raw, mean, std)
    X_eval_norm = apply_standardizer(X_eval_raw, mean, std)

    # 3. Đưa lên device dạng torch tensor
    X_tr = torch.tensor(X_tr_norm, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.int64, device=device)
    X_val = torch.tensor(X_val_norm, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.int64, device=device)
    X_eval = torch.tensor(X_eval_norm, dtype=torch.float32, device=device)
    y_eval_t = torch.tensor(y_eval_raw, dtype=torch.int64, device=device)

    # 4. In thông tin kích thước và baseline đoán lớp đa số
    majority_class = int(np.bincount(y_tr).argmax())
    majority_acc = float((y_val == majority_class).mean())

    print(f"Da chuan bi du lieu (device='{device}'):")
    print(f"  Train: X={tuple(X_tr.shape)}, y={tuple(y_tr_t.shape)}")
    print(f"  Val  : X={tuple(X_val.shape)}, y={tuple(y_val_t.shape)}")
    print(f"  Eval : X={tuple(X_eval.shape)}, y={tuple(y_eval_t.shape)}")
    print(f"  Lop da so trong train: {majority_class}")
    print(f"  Accuracy cua chien luoc 'luon doan lop da so' tren Val: {majority_acc:.4f} ({majority_acc*100:.2f}%)")

    return {
        "X_tr": X_tr,
        "y_tr": y_tr_t,
        "X_val": X_val,
        "y_val": y_val_t,
        "X_eval": X_eval,
        "y_eval": y_eval_t,
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    n = len(X)
    if isinstance(X, torch.Tensor):
        if shuffle:
            perm = torch.randperm(n, generator=generator, device=X.device)
        else:
            perm = torch.arange(n, device=X.device)
    else:
        if shuffle:
            perm = np.random.permutation(n)
        else:
            perm = np.arange(n)

    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        yield X[idx], y[idx]

