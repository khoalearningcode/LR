"""Trainer class encapsulating the training and validation loop (with checkpoint resume)."""
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence


class Trainer:
    """Encapsulates training, validation, and inference logic."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config,
        idx2char: Dict[int, str],
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.idx2char = idx2char
        self.device = config.DEVICE
        seed_everything(config.SEED, benchmark=getattr(config, "USE_CUDNN_BENCHMARK", False))

        # Loss and optimizer
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
        self.sr_criterion = nn.MSELoss()

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Scheduler: OneCycleLR can be resumed if state_dict is restored (requires same steps/epochs)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=config.EPOCHS,
        )
        self.scaler = GradScaler()

        # Tracking
        self.best_acc = 0.0
        self.current_epoch = 0  # next epoch index to run
        self.global_step = 0

    # ----------------------------
    # Paths
    # ----------------------------
    def _get_output_path(self, filename: str) -> str:
        output_dir = getattr(self.config, "OUTPUT_DIR", "results")
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)

    def _get_exp_name(self) -> str:
        return getattr(self.config, "EXPERIMENT_NAME", "baseline")

    def _get_ckpt_dir(self) -> str:
        out_dir = getattr(self.config, "OUTPUT_DIR", "results")
        ckpt_dirname = getattr(self.config, "CKPT_DIRNAME", "checkpoints")
        ckpt_dir = os.path.join(out_dir, ckpt_dirname)
        os.makedirs(ckpt_dir, exist_ok=True)
        return ckpt_dir

    def _ckpt_path(self, name: str) -> str:
        # name: "last" | "best" | "epoch_000" | ...
        return os.path.join(self._get_ckpt_dir(), f"{name}.pt")

    # ----------------------------
    # Checkpoint save/load (FULL STATE)
    # ----------------------------
    def save_checkpoint(self, name: str = "last", extra: Optional[dict] = None) -> str:
        """Save a FULL checkpoint (model+optimizer+scheduler+scaler+state)."""
        path = self._ckpt_path(name)

        # RNG states help reproducibility after resume
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                rng_state["cuda_all"] = torch.cuda.get_rng_state_all()
            except Exception:
                rng_state["cuda_all"] = None

        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch_next": int(self.current_epoch),
            "best_acc": float(self.best_acc),
            "global_step": int(self.global_step),
            "train_steps_per_epoch": int(len(self.train_loader)),
            "total_epochs": int(getattr(self.config, "EPOCHS", 0)),
            "rng": rng_state,
            "extra": extra or {},
        }

        torch.save(payload, path)
        return path

    def load_checkpoint(
        self,
        path: str,
        *,
        strict: bool = True,
        reset_optimizer: bool = False,
        reset_scheduler: bool = False,
        reset_scaler: bool = False,
    ) -> dict:
        """Load a FULL checkpoint. By default expects same steps_per_epoch and total_epochs for OneCycleLR."""
        ckpt = torch.load(path, map_location=self.device)

        # Model
        self.model.load_state_dict(ckpt["model"], strict=strict)

        # Epoch / tracking
        self.current_epoch = int(ckpt.get("epoch_next", 0))
        self.best_acc = float(ckpt.get("best_acc", 0.0))
        self.global_step = int(ckpt.get("global_step", 0))

        # Optim / scheduler / scaler
        if not reset_optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if not reset_scaler and "scaler" in ckpt:
            try:
                self.scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass

        # Scheduler resume safety (OneCycleLR)
        saved_spe = int(ckpt.get("train_steps_per_epoch", -1))
        saved_epochs = int(ckpt.get("total_epochs", -1))
        cur_spe = int(len(self.train_loader))
        cur_epochs = int(getattr(self.config, "EPOCHS", -1))

        if reset_scheduler:
            # keep current scheduler as-is
            pass
        else:
            if (saved_spe != -1 and saved_spe != cur_spe) or (saved_epochs != -1 and saved_epochs != cur_epochs):
                raise ValueError(
                    "Checkpoint scheduler mismatch for OneCycleLR.\n"
                    f" - saved steps_per_epoch={saved_spe}, current={cur_spe}\n"
                    f" - saved total_epochs={saved_epochs}, current={cur_epochs}\n"
                    "Fix: resume with the same dataset/batch-size/epochs, OR use --reset-scheduler."
                )
            if "scheduler" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler"])

        # Restore RNG states (optional)
        rng = ckpt.get("rng", None)
        if rng:
            try:
                random.setstate(rng.get("python", random.getstate()))
            except Exception:
                pass
            try:
                np.random.set_state(rng.get("numpy", np.random.get_state()))
            except Exception:
                pass
            try:
                torch.set_rng_state(rng.get("torch", torch.get_rng_state()))
            except Exception:
                pass
            if torch.cuda.is_available():
                cuda_all = rng.get("cuda_all", None)
                if cuda_all is not None:
                    try:
                        torch.cuda.set_rng_state_all(cuda_all)
                    except Exception:
                        pass

        return ckpt

    # ----------------------------
    # Weights-only save (compat)
    # ----------------------------
    def save_model(self, path: str = None) -> None:
        """Save weights-only checkpoint (best by default)."""
        if path is None:
            exp_name = self._get_exp_name()
            path = self._get_output_path(f"{exp_name}_best.pth")
        torch.save(self.model.state_dict(), path)

    def save_last_weights(self) -> str:
        exp_name = self._get_exp_name()
        path = self._get_output_path(f"{exp_name}_last.pth")
        torch.save(self.model.state_dict(), path)
        return path

    # ----------------------------
    # Train/Val
    # ----------------------------
    def train_one_epoch(self) -> float:
        """Train for one epoch (CTC loss in FP32 for stability)."""
        self.model.train()
        epoch_loss = 0.0

        save_every_steps = int(getattr(self.config, "SAVE_EVERY_STEPS", 0) or 0)
        use_sr = bool(getattr(self.config, "AUX_SR", False))
        sr_w = float(getattr(self.config, "SR_LOSS_W", 1.0))  # nếu chưa có thì mặc định 1.0
        grad_clip = float(getattr(self.config, "GRAD_CLIP", 5.0))

        pbar = tqdm(self.train_loader, desc=f"Ep {self.current_epoch + 1}/{self.config.EPOCHS}")
        for step, (images, hr_images, targets, target_lengths, _, _) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            hr_images = hr_images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # -------- Forward under AMP --------
            with autocast("cuda"):
                if use_sr:
                    preds, sr_out, grid = self.model(images, return_sr=True)
                else:
                    preds = self.model(images)
                    sr_out, grid = None, None

                sr_loss = None
                if use_sr and (sr_out is not None):
                    b, f, c, h, w = hr_images.size()
                    hr_target = hr_images.view(b * f, c, h, w)
                    if grid is not None:
                        hr_target = torch.nn.functional.grid_sample(hr_target, grid, align_corners=False)
                    sr_loss = self.sr_criterion(sr_out, hr_target)
            # -----------------------------------

            # -------- CTC loss in FP32 (IMPORTANT) --------
            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=preds.size(1),
                dtype=torch.long,
                device=targets.device,
            )
            ctc_loss = self.criterion(preds.float().permute(1, 0, 2), targets, input_lengths, target_lengths)
            loss = ctc_loss
            if use_sr and (sr_loss is not None) and (sr_w > 0):
                loss = loss + sr_w * sr_loss.float()
            # ---------------------------------------------

            # Skip non-finite to avoid collapse
            if not torch.isfinite(loss):
                pbar.set_postfix({
                    "loss": "nonfinite",
                    "lr": float(self.scheduler.get_last_lr()[0]),
                    "sc": float(self.scaler.get_scale()),
                })
                self.optimizer.zero_grad(set_to_none=True)
                continue

            # Backward
            self.scaler.scale(loss).backward()

            # Clip grad (log grad norm)
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            # Optim step (AMP-safe)
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Step scheduler only if optimizer stepped
            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()

            epoch_loss += float(loss.item())
            self.global_step += 1

            # Progress log
            postfix = {
                "loss": float(loss.item()),
                "lr": float(self.scheduler.get_last_lr()[0]),
                "gn": float(grad_norm),
                "sc": float(self.scaler.get_scale()),
            }
            if use_sr and (sr_loss is not None):
                postfix["ctc"] = float(ctc_loss.item())
                postfix["sr"] = float(sr_loss.item())
            pbar.set_postfix(postfix)

            # Optional mid-epoch save
            if save_every_steps > 0 and (self.global_step % save_every_steps == 0):
                try:
                    self.save_checkpoint("last")
                except Exception:
                    pass

        return epoch_loss / max(1, len(self.train_loader))


    def validate(self) -> Tuple[Dict[str, float], List[str]]:
        if self.val_loader is None:
            return {"loss": 0.0, "acc": 0.0}, []

        self.model.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0
        submission_data: List[str] = []

        with torch.no_grad():
            for images, _, targets, target_lengths, labels_text, track_ids in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                preds = self.model(images)

                input_lengths = torch.full(
                    (images.size(0),),
                    preds.size(1),
                    dtype=torch.long,
                    device=targets.device,
                )
                loss = self.criterion(preds.float().permute(1, 0, 2), targets, input_lengths, target_lengths)
                val_loss += float(loss.item())

                decoded_list = decode_with_confidence(preds, self.idx2char)

                for i, (pred_text, conf) in enumerate(decoded_list):
                    gt_text = labels_text[i]
                    track_id = track_ids[i]
                    if pred_text == gt_text:
                        total_correct += 1
                    submission_data.append(f"{track_id},{pred_text};{conf:.4f}")

                total_samples += len(labels_text)

        avg_val_loss = val_loss / max(1, len(self.val_loader))
        val_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
        metrics = {"loss": avg_val_loss, "acc": val_acc}
        return metrics, submission_data

    def save_submission(self, submission_data: List[str]) -> None:
        exp_name = self._get_exp_name()
        filename = self._get_output_path(f"submission_{exp_name}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(submission_data))
        print(f"📝 Saved {len(submission_data)} lines to {filename}")

    def fit(self) -> None:
        print(f"🚀 TRAINING START | Device: {self.device} | Epochs: {self.config.EPOCHS}")
        start_epoch = int(getattr(self, "current_epoch", 0))

        save_every_epochs = int(getattr(self.config, "SAVE_EVERY_EPOCHS", 1) or 1)

        for epoch in range(start_epoch, self.config.EPOCHS):
            self.current_epoch = epoch

            avg_train_loss = self.train_one_epoch()

            val_metrics, submission_data = self.validate()
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["acc"]
            current_lr = self.scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch + 1}/{self.config.EPOCHS}: "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.2e}"
            )

            # Save best
            if self.val_loader is not None and val_acc > self.best_acc:
                self.best_acc = float(val_acc)
                self.save_model()  # weights-only best
                best_full = self.save_checkpoint("best", extra={"val_acc": float(val_acc), "val_loss": float(val_loss)})
                exp_name = self._get_exp_name()
                best_weights = self._get_output_path(f"{exp_name}_best.pth")
                print(f"  ⭐ Saved Best: {best_weights} | {best_full}")

                if submission_data:
                    self.save_submission(submission_data)

            # Always save last (weights-only + full state)
            self.current_epoch = epoch + 1  # next epoch index
            self.save_last_weights()
            if save_every_epochs > 0 and ((epoch + 1) % save_every_epochs == 0):
                self.save_checkpoint("last", extra={"epoch": epoch + 1, "val_acc": float(val_acc), "val_loss": float(val_loss)})

        # If submission-mode (no val), treat final as best
        if self.val_loader is None:
            self.save_model()
            self.save_checkpoint("best")
            exp_name = self._get_exp_name()
            model_path = self._get_output_path(f"{exp_name}_best.pth")
            print(f"  💾 Saved final model: {model_path}")

        print(f"\n✅ Training complete! Best Val Acc: {self.best_acc:.2f}%")

    # ----------------------------
    # Inference helpers
    # ----------------------------
    def predict(self, loader: DataLoader) -> List[Tuple[str, str, float]]:
        self.model.eval()
        results: List[Tuple[str, str, float]] = []
        with torch.no_grad():
            for images, _, _, _, _, track_ids in loader:  # collate returns 6 items
                images = images.to(self.device)
                preds = self.model(images)
                decoded_list = decode_with_confidence(preds, self.idx2char)
                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append((track_ids[i], pred_text, conf))
        return results

    def predict_test(self, test_loader: DataLoader, output_filename: str = "submission_final.txt") -> None:
        print("🔮 Running inference on test data...")
        results: List[str] = []
        self.model.eval()

        with torch.no_grad():
            for images, _, _, _, _, track_ids in tqdm(test_loader, desc="Test Inference"):
                images = images.to(self.device)
                preds = self.model(images)
                decoded_list = decode_with_confidence(preds, self.idx2char)
                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append(f"{track_ids[i]},{pred_text};{conf:.4f}")

        out_path = self._get_output_path(output_filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(results))
        print(f"✅ Saved {len(results)} lines to {out_path}") 