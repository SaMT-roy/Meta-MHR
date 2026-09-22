import argparse
import importlib
import math
import pathlib
import time
from collections import Counter

import torch
from torch.utils.data import DataLoader

import losses
from dataset import PoseData, split_clips
from prepare import ik

HERE = pathlib.Path(__file__).resolve().parent

MODELS = {"spatial": "model_spatial", "iterative": "model_iterative"}

BATCH = 128
WORKERS = 0
EPOCHS = 30
FREEZE_EPOCHS = 1
BACKBONE_LR = 3e-4
HEAD_LR = 3e-4
SEED = 0


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def build_optimiser(net):
    backbone = list(net.backbone.parameters())
    head = [p for name, p in net.named_parameters() if not name.startswith("backbone.")]
    return torch.optim.AdamW(
        [{"params": backbone, "lr": 0.0}, {"params": head, "lr": 0.0}],
        weight_decay=1e-4)

def set_learning_rate(optimiser, step, frozen_steps, total_steps):
    if step < frozen_steps:
        backbone = 0.0
        head = HEAD_LR * (step + 1) / frozen_steps
    else:
        progress = (step - frozen_steps) / max(1, total_steps - frozen_steps)
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        backbone = BACKBONE_LR * decay
        head = HEAD_LR * decay

    optimiser.param_groups[0]["lr"] = backbone
    optimiser.param_groups[1]["lr"] = head
    return backbone, head


def set_backbone_frozen(net, frozen):
    net.backbone.requires_grad_(not frozen)


def predict(net, batch, device):
    out = net(batch["image"].to(device), batch["context"].to(device))
    return {name: value.cpu() for name, value in out.items()}


def average_over(loader, score):
    totals, seen = Counter(), 0
    for batch in loader:
        count = len(batch["image"])
        for name, value in score(batch).items():
            totals[name] += value * count
        seen += count
    return {name: total / seen for name, total in totals.items()}


def train_one_epoch(net, loader, body, optimiser, device, frozen, step, frozen_steps, total_steps):
    net.train()
    if frozen:
        net.backbone.eval()

    total = 0.0
    rates = (0.0, 0.0)
    for batch in loader:
        rates = set_learning_rate(optimiser, step, frozen_steps, total_steps)
        loss, _parts = losses.compute_loss(predict(net, batch, device), batch, body)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        total += float(loss.detach())
        step += 1
    return total / len(loader), step, rates


def evaluate(net, loader, body, device):
    net.eval()

    def score(batch):
        with torch.no_grad():
            prediction = predict(net, batch, device)
        loss, _parts = losses.compute_loss(prediction, batch, body)
        return {"loss": float(loss), **losses.metrics(prediction, batch, body)}

    return average_over(loader, score)


def floor(loader, body):
    return average_over(loader, lambda batch: losses.metrics(
        losses.labels_as_prediction(batch), batch, body))


def main(model_name="spatial", epochs=EPOCHS, backbone="resnet18"):
    device = pick_device()
    torch.manual_seed(SEED)

    model = importlib.import_module(MODELS[model_name])

    train_clips, val_clips = split_clips()
    train_data = PoseData(train_clips, augment=True)
    val_data = PoseData(val_clips, augment=False)

    train_loader = DataLoader(train_data, BATCH, shuffle=True,
                              num_workers=WORKERS, drop_last=True)
    val_loader = DataLoader(val_data, BATCH, num_workers=WORKERS)

    body = losses.Body(ik.Rig())
    net = model.PoseNet(backbone, average=model.average_target(train_data)).to(device)
    optimiser = build_optimiser(net)

    frozen_steps = FREEZE_EPOCHS * len(train_loader)
    total_steps = epochs * len(train_loader)

    checkpoint = HERE / f"{MODELS[model_name]}.pt"
    parameters = sum(p.numel() for p in net.parameters()) / 1e6

    print(f"device {device}   {model_name} / {backbone}   {parameters:.1f} M parameters")
    print(f"train {len(train_clips):2d} clips {len(train_data):5d} frames"
          f"    held back {len(val_clips):2d} clips {len(val_data):5d} frames")
    print(f"backbone frozen for {FREEZE_EPOCHS} epochs while the head warms up to {HEAD_LR:g}")

    step, best = 0, float("inf")
    for epoch in range(1, epochs + 1):
        started = time.time()

        frozen = epoch <= FREEZE_EPOCHS
        set_backbone_frozen(net, frozen)

        train_loss, step, rates = train_one_epoch(
            net, train_loader, body, optimiser, device, frozen, step, frozen_steps, total_steps)
        scores = evaluate(net, val_loader, body, device)

        saved = scores["loss"] < best
        if saved:
            best = scores["loss"]
            torch.save({"model": model_name, "backbone": backbone,
                        "weights": net.state_dict()}, checkpoint)

        print(f"epoch {epoch:3d}   lr {rates[0]:.1e}/{rates[1]:.1e}"
              f"   train {train_loss:6.3f}   val {scores['loss']:6.3f}"
              f"   {scores['mpjpe_mm']:5.0f} mm {scores['pixel_error']:6.1f} px"
              f" {scores['rotation_deg']:5.1f} deg   {time.time() - started:4.0f} s"
              f"{'   saved' if saved else ''}")

    final_scores = evaluate(net, val_loader, body, device)
    floor_scores = floor(val_loader, body)

    print(f"\n{'':16s}{'final':>10s}{'floor':>10s}")
    for name in ("mpjpe_mm", "pixel_error", "root_error_cm", "rotation_deg"):
        print(f"{name:16s}{final_scores[name]:10.2f}{floor_scores[name]:10.2f}")
    print(f"\nsaved to {checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="spatial", choices=list(MODELS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--backbone", default="resnet18")
    args = parser.parse_args()
    main(args.model, args.epochs, args.backbone)
