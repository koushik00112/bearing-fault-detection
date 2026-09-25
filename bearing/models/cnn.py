"""Small 1D CNN on raw multichannel windows.

Architecture in the spirit of WDCNN (Zhang et al., 2017): a wide, strided first
convolution acts as a learned filter bank, followed by narrow convolutions and global
average pooling. Small enough to train on a laptop CPU.
"""

import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from bearing.windowing import WindowSet

log = logging.getLogger(__name__)


class Net(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, width: int = 16) -> None:
        super().__init__()

        def block(cin: int, cout: int, k: int, stride: int = 1) -> list[nn.Module]:
            return [
                nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2),
                nn.BatchNorm1d(cout),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]

        self.features = nn.Sequential(
            *block(in_channels, width, 64, stride=8),
            *block(width, 2 * width, 3),
            *block(2 * width, 4 * width, 3),
            *block(4 * width, 4 * width, 3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(4 * width, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def pick_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CNNModel:
    name = "cnn"

    def __init__(self, classes: tuple[str, ...], fs: float, **params: Any) -> None:
        self.classes = tuple(classes)
        self.fs = fs
        self.params = {
            "epochs": 20,
            "batch_size": 128,
            "lr": 2e-3,
            "width": 16,
            "patience": 4,
            "device": "auto",
            **params,
        }
        self.net: Net | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.channels: tuple[str, ...] = ()
        self.history: list[dict[str, float]] = []

    def _norm(self, X: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None
        return ((np.asarray(X, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def fit(self, train: WindowSet, val: WindowSet | None, seed: int) -> None:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        device = pick_device(self.params["device"])
        self.channels = train.channels
        # Per-channel statistics from the training windows only.
        sample = np.asarray(train.X[rng.choice(len(train), min(len(train), 2000), replace=False)])
        self.mean = sample.mean(axis=(0, 2), keepdims=True)[0]
        self.std = sample.std(axis=(0, 2), keepdims=True)[0] + 1e-6

        y = np.array([self.classes.index(label) for label in train.meta["label"]])
        counts = np.bincount(y, minlength=len(self.classes)).astype(np.float32)
        weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(self.classes))

        net = Net(train.X.shape[1], len(self.classes), self.params["width"]).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.params["lr"], weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
        bs = self.params["batch_size"]

        best_state, best_loss, bad_epochs = None, float("inf"), 0
        for epoch in range(self.params["epochs"]):
            net.train()
            order = rng.permutation(len(train))
            total = 0.0
            for i in range(0, len(order), bs):
                idx = np.sort(order[i : i + bs])  # sorted: faster reads from a memmap
                xb = torch.from_numpy(self._norm(train.X[idx])).to(device)
                yb = torch.from_numpy(y[idx]).to(device)
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            record = {"epoch": epoch, "train_loss": total / len(train)}
            if val is not None and len(val):
                self.net = net
                vy = np.array([self.classes.index(label) for label in val.meta["label"]])
                p = self.predict_proba(val.X)
                record["val_loss"] = float(-np.mean(np.log(p[np.arange(len(vy)), vy] + 1e-9)))
                if record["val_loss"] < best_loss - 1e-4:
                    best_loss, bad_epochs = record["val_loss"], 0
                    best_state = copy.deepcopy(net.state_dict())
                else:
                    bad_epochs += 1
            self.history.append(record)
            log.info("cnn epoch %s", record)
            if val is not None and bad_epochs >= self.params["patience"]:
                break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net = net.to("cpu")

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, batch: int = 512) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("model is not fitted")
        self.net.eval()
        device = next(self.net.parameters()).device
        out = []
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(self._norm(X[i : i + batch])).to(device)
            out.append(torch.softmax(self.net(xb), dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, len(self.classes)))

    def n_parameters(self) -> int:
        assert self.net is not None
        return sum(p.numel() for p in self.net.parameters())

    def save(self, directory: Path) -> None:
        assert self.net is not None and self.mean is not None and self.std is not None
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), directory / "model.pt")
        np.savez(directory / "norm.npz", mean=self.mean, std=self.std)
        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "model": "cnn",
                    "classes": list(self.classes),
                    "fs": self.fs,
                    "channels": list(self.channels),
                    "params": self.params,
                    "n_parameters": self.n_parameters(),
                    "history": self.history,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, directory: Path) -> "CNNModel":
        meta = json.loads((directory / "meta.json").read_text())
        model = cls(tuple(meta["classes"]), meta["fs"], **meta["params"])
        model.channels = tuple(meta["channels"])
        norm = np.load(directory / "norm.npz")
        model.mean, model.std = norm["mean"], norm["std"]
        model.net = Net(len(model.channels), len(model.classes), meta["params"]["width"])
        model.net.load_state_dict(torch.load(directory / "model.pt", weights_only=True))
        model.net.eval()
        return model
