"""train.py — IMPLEMENTATION. Bạn phải tự hoàn thiện mọi hàm có `unfinished placeholder`.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.1,
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    denom = 2 * tp + (cm.sum(axis=0) - tp) + (cm.sum(axis=1) - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom != 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    model.eval()
    chunks = [model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)]
    if not chunks:
        return torch.empty(0, dtype=torch.int64, device=X.device)
    return torch.cat(chunks).to(torch.int64)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    model.eval()
    n, loss_sum = len(X), 0.0
    cm = torch.zeros((7, 7), dtype=torch.int64, device=y.device)
    for start in range(0, n, batch_size):
        logits = model(X[start:start + batch_size])
        target = y[start:start + batch_size]
        loss_sum += float(compute_loss(logits, target, loss_name).item()) * len(target)
        pred = logits.argmax(dim=1)
        cm += torch.bincount(target * 7 + pred, minlength=49).reshape(7, 7)
    cm_np = cm.cpu().numpy()
    correct = int(np.trace(cm_np))
    return {"loss": loss_sum / max(1, n), "acc": correct / max(1, n),
            "macro_f1": macro_f1_from_confusion(cm_np), "confusion_matrix": cm_np}


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "mse":
        targets = F.one_hot(y.long(), num_classes=logits.shape[1]).to(dtype=logits.dtype)
        return F.mse_loss(logits, targets)
    raise ValueError("loss_name must be 'ce' or 'mse'")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    if cfg["lr"] is None or cfg["batch"] <= 0 or cfg["epochs"] <= 0:
        raise ValueError("cfg requires a positive lr, batch, and epochs")
    set_seed(int(cfg["seed"]))
    X_tr, y_tr, X_val, y_val = (data[k] for k in ("X_tr", "y_tr", "X_val", "y_val"))
    device = X_tr.device
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=float(cfg["dropout"]), init=cfg["init"]).to(device)
    expected = EXPECTED_PARAMS.get(tuple(cfg["hidden"]))
    if expected is not None and count_params(model) != expected:
        raise AssertionError(f"parameter count {count_params(model)} != expected {expected}")
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), float(cfg["lr"]),
                                float(cfg["weight_decay"]), float(cfg["momentum"]))
    precision = cfg["precision"]
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16, or bf16")
    use_amp = device.type == "cuda" and precision != "fp32"
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and precision == "fp16")
    generator = torch.Generator(device=device).manual_seed(int(cfg["seed"]))
    history = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "grad_norm", "epoch_time_s")}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    step0_loss = evaluate(model, X_val, y_val, cfg["loss"])["loss"]
    best_loss, best_epoch, best_state, best_metrics = math.inf, 0, None, None
    diverged = False
    for epoch in range(1, int(cfg["epochs"]) + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        model.train()
        grad_norms, batches = [], 0
        for xb, yb in iterate_batches(X_tr, y_tr, int(cfg["batch"]), generator=generator, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = compute_loss(model(xb), yb, cfg["loss"])
            if not torch.isfinite(loss):
                diverged = True
                break
            scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norms.append(clip_gradients(model.parameters(), cfg["clip_norm"]))
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            batches += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time
        if diverged:
            break
        train_metrics = evaluate(model, X_tr, y_tr, cfg["loss"])
        val_metrics = evaluate(model, X_val, y_val, cfg["loss"])
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["grad_norm"].append(float(np.mean(grad_norms)) if grad_norms else 0.0)
        history["epoch_time_s"].append(elapsed)
        if val_metrics["loss"] < best_loss:
            best_loss, best_epoch = val_metrics["loss"], epoch
            best_metrics, best_state = val_metrics, copy.deepcopy(model.state_dict())
    summary = {
        "step0_loss": step0_loss,
        "best_val_loss": best_loss if best_metrics else None,
        "best_epoch": best_epoch,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "val_acc": best_metrics["acc"] if best_metrics else None,
        "val_macro_f1": best_metrics["macro_f1"] if best_metrics else None,
        "time_per_epoch_s": float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else None,
        "peak_mem_MB": torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None,
        "diverged": diverged,
    }
    return {"cfg": cfg, "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    row_ids = np.asarray(row_id, dtype=np.int64)
    predictions = np.asarray(preds, dtype=np.int64)
    if row_ids.ndim != 1 or predictions.shape != row_ids.shape or len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("row_id và preds phải là vector cùng chiều dài, row_id không trùng")
    if len(predictions) and (predictions.min() < 0 or predictions.max() > 6):
        raise ValueError("predicted labels must be in [0, 6]")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("row_id", "pred"))
        writer.writerows(zip(row_ids.tolist(), predictions.tolist()))


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=float(cfg["dropout"]), init=cfg["init"]).to(data["X_eval"].device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
