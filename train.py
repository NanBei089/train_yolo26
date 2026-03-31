from __future__ import annotations

from pathlib import Path

import tensorflow as tf
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO


def _resolve_data_yaml(data_yaml: str) -> str:
    """优先使用传入 data.yaml，不存在则回退到 split_dataset.py 生成的 dataset.yaml。"""
    preferred = Path(data_yaml)
    if preferred.exists():
        return str(preferred)

    fallback = Path("./dataset/dataset.yaml")
    if fallback.exists():
        print(f"未找到 {preferred}，自动使用 {fallback}")
        return str(fallback)

    raise FileNotFoundError(
        f"数据集配置文件不存在：{preferred}，且未找到 {fallback}"
    )


def _resolve_device(device: int | str) -> int | str:
    """优先用 GPU，GPU 不可用时自动回退 CPU。"""
    if isinstance(device, str) and device.lower() == "cpu":
        return "cpu"
    if device == 0 and not torch.cuda.is_available():
        print("检测到 CUDA 不可用，自动切换到 CPU")
        return "cpu"
    return device


def _create_tf_writer(run_name: str):
    """创建 TensorFlow 日志写入器，用于训练过程监测。"""
    log_dir = Path("runs/tf_monitor") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"TensorFlow 版本: {tf.__version__}")
    print(f"TensorFlow GPU 设备数: {len(tf.config.list_physical_devices('GPU'))}")
    print(f"TensorFlow 日志目录: {log_dir}")
    return tf.summary.create_file_writer(str(log_dir))


def _register_tf_callback(model: YOLO, writer) -> None:
    """将训练损失和验证指标写入 TensorFlow 日志。"""

    def on_fit_epoch_end(trainer) -> None:
        epoch = int(getattr(trainer, "epoch", 0)) + 1

        with writer.as_default():
            tloss = getattr(trainer, "tloss", None)
            if tloss is not None:
                values = tloss.tolist() if hasattr(tloss, "tolist") else list(tloss)
                names = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
                for idx, value in enumerate(values[:3]):
                    tf.summary.scalar(names[idx], float(value), step=epoch)

            metrics = getattr(trainer, "metrics", {}) or {}
            for key, value in metrics.items():
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    continue
                safe_key = str(key).replace("/", "_").replace("(", "").replace(")", "")
                tf.summary.scalar(f"val/{safe_key}", scalar, step=epoch)

            writer.flush()

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def _register_progress_callback(model: YOLO, total_epochs: int) -> None:
    """使用 tqdm 展示训练进度。"""
    epoch_bar = tqdm(total=total_epochs, desc="Training", unit="epoch")

    def on_fit_epoch_end(trainer) -> None:
        current_epoch = int(getattr(trainer, "epoch", 0)) + 1
        tloss = getattr(trainer, "tloss", None)
        loss_text = ""
        if tloss is not None:
            values = tloss.tolist() if hasattr(tloss, "tolist") else list(tloss)
            if values:
                loss_text = f"loss={sum(values[:3]):.4f}"
        epoch_bar.set_postfix_str(f"epoch {current_epoch}/{total_epochs} {loss_text}".strip())
        epoch_bar.update(1)

    def on_train_end(trainer) -> None:
        if epoch_bar.n < epoch_bar.total:
            epoch_bar.update(epoch_bar.total - epoch_bar.n)
        epoch_bar.close()

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)


def train_yolo(
    model_name: str,
    data_yaml: str,
    batch_size: int = -1,
    epochs: int = 180,
    img_size: int = 640,
    device: int | str = 0,
    workers: int = 4,
) -> None:
    # 加载 YOLO 模型
    model = YOLO(model_name)

    # 定义数据增强配置
    augmentations = {
        "hsv_h": 0.012,    # 色相增强范围
        "hsv_s": 0.5,      # 饱和度增强范围
        "hsv_v": 0.3,      # 明度增强范围
        "degrees": 3.0,    # 随机旋转的最大角度
        "translate": 0.05, # 随机平移范围
        "scale": 0.3,      # 随机缩放范围
        "shear": 0.0,      # 随机剪切范围
        "flipud": 0.0,     # 随机上下翻转
        "fliplr": 0.2,     # 随机左右翻转
    }

    # 启用早停（EarlyStopping）
    early_stopping = True
    patience = 30 if early_stopping else epochs

    run_name = "yolo26s-vehicle-opt"
    writer = _create_tf_writer(run_name)
    _register_tf_callback(model, writer)
    _register_progress_callback(model, epochs)

    resolved_data_yaml = _resolve_data_yaml(data_yaml)
    resolved_device = _resolve_device(device)

    # 设置训练参数
    # 注意：Ultralytics 中不存在 augmentations/early_stopping/scheduler 这些字段名，
    # 这里已映射为等效参数（hsv_*/patience/cos_lr）。
    try:
        model.train(
            data=resolved_data_yaml,       # 数据集配置文件
            epochs=epochs,                 # 训练周期数
            imgsz=img_size,                # 图像大小
            batch=batch_size,              # 批次大小（-1 为 AutoBatch）
            device=resolved_device,        # 使用 GPU 编号或 'cpu'
            name=run_name,                 # 训练保存路径名称
            workers=workers,               # 加载数据的线程数
            save_period=10,                # 每 10 个周期保存一次模型
            cache=True,                    # 小数据集建议缓存，提高训练速度
            optimizer="auto",              # 自动选择优化器与学习率
            cos_lr=True,                   # 余弦学习率调度
            patience=patience,             # 早停耐心次数
            close_mosaic=10,               # 最后 10 轮关闭 Mosaic 稳定收敛
            project="runs/train",
            exist_ok=True,
            hsv_h=augmentations["hsv_h"],
            hsv_s=augmentations["hsv_s"],
            hsv_v=augmentations["hsv_v"],
            degrees=augmentations["degrees"],
            translate=augmentations["translate"],
            scale=augmentations["scale"],
            shear=augmentations["shear"],
            flipud=augmentations["flipud"],
            fliplr=augmentations["fliplr"],
        )
    finally:
        writer.close()

    print(f"训练完成，结果保存在：runs/train/{run_name}")
    print(f"TensorFlow 监测日志：runs/tf_monitor/{run_name}")


def main() -> None:
    # 自定义训练模型参数
    model_name = "yolo26s.pt"
    data_yaml = "./dataset/dataset.yaml"
    batch_size = -1
    epochs = 180
    img_size = 640
    device = 0
    workers = 4

    # 开始训练
    train_yolo(model_name, data_yaml, batch_size, epochs, img_size, device, workers)


if __name__ == "__main__":
    main()
