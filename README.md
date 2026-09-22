# YOLOv8 Object Detection Example

PyTorch YOLOv8-style object-detection training, validation, interactive inference, and ONNX export example. The implementation includes mosaic and geometric augmentation, rectangular validation batches, EMA, early stopping, checkpointing, NMS, and optional ONNX simplification.

This repository is a training example, not a released detector or board deliverable. Dataset classes, metrics, and deployment conversion must be defined by the target project.

## Requirements

The scripts use Python 3 with PyTorch, torchvision, NumPy, OpenCV, Pillow, tqdm, Loguru, Albumentations, ONNX, ONNX Simplifier, THOP, and optionally Weights & Biases.

## Dataset

Use YOLO image/label pairs with matching relative paths and filenames:

```text
dataset/
├── images/
│   ├── train/example.jpg
│   └── val/example.jpg
└── labels/
    ├── train/example.txt
    └── val/example.txt
```

Each label row uses normalized YOLO format:

```text
class_id center_x center_y width height
```

The default class order is defined by `DetectionDataset.class_names` in `dataset.py`:

```text
bicycle, door_warning_sign, electric_bicycle, gastank, people
```

Edit that list for a different dataset and pass the same ordered values through `trainer.py --class_names`. The CLI argument sets model/output labels but does not replace the dataset class list.

## Train

```bash
python3 trainer.py \
  --device_id cuda \
  --trainset_path <dataset>/images/train \
  --valset_path <dataset>/images/val \
  --class_names bicycle door_warning_sign electric_bicycle gastank people \
  --pretrained_pt_path res/pvdetection_07fe2b.pt \
  --batch_size 16 \
  --image_size 640 \
  --max_epochs 300 \
  --ema_enabled \
  --early_stop
```

Add `--wandb_enabled` for offline Weights & Biases logging. Checkpoints are written under `--save_dir` (default `runs/`); the best checkpoint is `<model_name>.pt` and the latest checkpoint is `<model_name>_latest.pt`.

Use `--debug_mode` for a one-iteration-per-epoch pipeline check. It is not a training-quality validation.

## Run Interactive Inference

Camera input:

```bash
python3 demo.py --weight runs/<model>.pt --device cuda --camera
```

The demo also supports `--screenshot` and `--h264 <stream-path>`. Configure `--conf_thr`, `--iou_thr`, and `--size` to match the evaluation or deployment contract.

## Export ONNX

```bash
python3 export.py \
  --weight runs/<model>.pt \
  --device cpu \
  --input_shape 1 3 640 640 \
  --input_names image \
  --output_names head_0_box head_0_cls head_1_box head_1_cls head_2_box head_2_cls \
  --opset_version 13 \
  --enable_onnxsim
```

The exporter enables the model's porting output mode, prints parameter and FLOP estimates, and waits for confirmation before writing `<model>.onnx` beside the checkpoint.

## Verification

- Validate image/label pairing and class IDs before training.
- Record precision, recall, `mAP@0.5`, and `mAP@0.5:0.95` from the validation split.
- Compare the ONNX outputs with the PyTorch checkpoint on fixed samples before deployment conversion.
- Record input size, class order, checkpoint hash, and confidence/NMS thresholds with reported results.

No accuracy target or target-chip compatibility is asserted by this repository.

## Files

```text
model.py              detector definition
dataset.py            YOLO dataset and augmentation pipeline
trainer.py            training and validation entry point
demo.py               interactive inference entry point
export.py             ONNX export entry point
box.py, nms.py        box transforms and non-maximum suppression
```
