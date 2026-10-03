"""plots.py — IMPLEMENTATION. Bạn phải tự hoàn thiện mọi hàm có `unfinished placeholder`.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    cfg, history = result["cfg"], result["history"]
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["val_acc"], label="accuracy")
    axes[1].plot(epochs, history["val_macro_f1"], label="macro-F1")
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score")
    axes[1].legend()
    axes[2].plot(epochs, history["grad_norm"], label="pre-clip norm")
    axes[2].set(title="Gradient norm", xlabel="Epoch", ylabel="L2 norm")
    axes[2].legend()
    best_epoch = result["summary"].get("best_epoch")
    if best_epoch in epochs:
        for ax in axes:
            ax.axvline(best_epoch, color="black", linestyle="--", alpha=0.5)
    fig.suptitle(f"{cfg.get('exp_id')} | {cfg.get('optimizer')} lr={cfg.get('lr')} batch={cfg.get('batch')}")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    if not results:
        raise ValueError("results must not be empty")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for item in results:
        hist = item["history"]
        if metric not in hist:
            raise KeyError(f"metric {metric!r} not present in history")
        ax.plot(hist["epoch"], hist[metric], marker=".", label=item["cfg"]["exp_id"])
    ax.set(title=title or f"Comparison: {metric}", xlabel="Epoch", ylabel=metric)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
