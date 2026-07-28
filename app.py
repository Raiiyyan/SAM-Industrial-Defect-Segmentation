"""
app.py — Production Tkinter desktop app for IndustrialSAM inference.

Features:
  - Open image file (browse)
  - Select dataset profiler (dropdown)
  - Run Inspection button (background thread)
  - Side-by-side Matplotlib canvases (raw + overlay)
  - Dynamic status bar with class prediction + confidence
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import torch
import torch.nn.functional as F

from model import IndustrialSAM
from classifier_head import CLASS_VOCAB

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

CLASS_NAMES = [CLASS_VOCAB[i] for i in range(len(CLASS_VOCAB))]

DATASETS = [
    "MVTec AD",
    "MVTec AD 2",
    "Severstal Steel",
    "NEU Surface Defect",
    "DAGM2007",
    "DefectSpectrum",
]


class IndustrialSAMApp:
    def __init__(self, root, checkpoint_path, device="cpu"):
        self.root = root
        self.root.title("IndustrialSAM — Defect Inspection")
        self.root.geometry("1100x600")
        self.root.minsize(900, 500)

        self.device = device
        self.current_image_path = None
        self.is_running = False

        # Load model once at startup
        self.status_var = tk.StringVar(value="Loading model ...")
        self._build_ui()
        self.root.after(100, lambda: self._load_model(checkpoint_path))

    def _build_ui(self):
        # Top toolbar
        toolbar = ttk.Frame(self.root, padding=5)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Open Image File", command=self._open_file).pack(
            side=tk.LEFT, padx=5
        )

        ttk.Label(toolbar, text="Dataset:").pack(side=tk.LEFT, padx=(15, 5))
        self.dataset_var = tk.StringVar(value=DATASETS[0])
        dataset_menu = ttk.Combobox(
            toolbar,
            textvariable=self.dataset_var,
            values=DATASETS,
            state="readonly",
            width=20,
        )
        dataset_menu.pack(side=tk.LEFT, padx=5)

        self.run_btn = ttk.Button(
            toolbar,
            text="Run Inspection",
            command=self._run_inspection,
            state=tk.DISABLED,
        )
        self.run_btn.pack(side=tk.LEFT, padx=15)

        # Canvas area (two matplotlib panels side by side)
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig_left, self.ax_left = plt.subplots(figsize=(5, 4))
        self.ax_left.set_title("Input Image")
        self.ax_left.axis("off")
        self.canvas_left = FigureCanvasTkAgg(self.fig_left, master=canvas_frame)
        self.canvas_left.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig_right, self.ax_right = plt.subplots(figsize=(5, 4))
        self.ax_right.set_title("Prediction Overlay")
        self.ax_right.axis("off")
        self.canvas_right = FigureCanvasTkAgg(self.fig_right, master=canvas_frame)
        self.canvas_right.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Status bar
        self.status_label = ttk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Arial", 12, "bold"),
            anchor=tk.CENTER,
            padding=10,
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Detail bar (all class probabilities)
        self.detail_var = tk.StringVar(value="")
        self.detail_label = ttk.Label(
            self.root,
            textvariable=self.detail_var,
            font=("Consolas", 9),
            anchor=tk.CENTER,
            padding=3,
        )
        self.detail_label.pack(side=tk.BOTTOM, fill=tk.X)

    def _load_model(self, checkpoint_path):
        try:
            # Load SAM base weights first, then overlay trained weights
            self.model = IndustrialSAM(
                "sam_vit_b_01ec64.pth", device=self.device, train_resolution=512
            )
            # Load trained weights from checkpoint
            import torch
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            self.model.eval()
            self.status_var.set("Model loaded — open an image to begin inspection")
            self.run_btn.config(state=tk.NORMAL)
        except Exception as e:
            self.status_var.set(f"Model load FAILED: {e}")
            messagebox.showerror("Model Error", f"Failed to load checkpoint:\n{e}")

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        if not os.path.isfile(path):
            messagebox.showwarning("File Error", "Selected file does not exist.")
            return

        self.current_image_path = path
        self._display_image(path)
        self.status_var.set(f"Loaded: {os.path.basename(path)} — click Run Inspection")

    def _display_image(self, path):
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            messagebox.showwarning("Image Error", f"Cannot read image:\n{path}")
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        self.ax_left.clear()
        self.ax_left.imshow(img_rgb)
        self.ax_left.set_title("Input Image")
        self.ax_left.axis("off")
        self.canvas_left.draw()

        self.ax_right.clear()
        self.ax_right.imshow(img_rgb)
        self.ax_right.set_title("Prediction Overlay")
        self.ax_right.axis("off")
        self.canvas_right.draw()

        self._raw_image = img_rgb

    def _run_inspection(self):
        if self.is_running:
            return
        if self.current_image_path is None:
            self.status_var.set("WARNING: No image loaded — select an image first")
            return
        if not hasattr(self, "model"):
            self.status_var.set("WARNING: Model not loaded yet")
            return

        self.is_running = True
        self.run_btn.config(state=tk.DISABLED)
        self.status_var.set("Processing ...")
        threading.Thread(target=self._inference_thread, daemon=True).start()

    def _inference_thread(self):
        try:
            img_bgr = cv2.imread(self.current_image_path, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(
                img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR
            )

            # Preprocess
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            # Auto-generate box from image content (find non-black region)
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Use bounding box of largest contour
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                # Scale to 512 space
                h_img, w_img = img_rgb.shape[:2]
                x1 = int(x / w_img * 511)
                y1 = int(y / h_img * 511)
                x2 = int((x + w) / w_img * 511)
                y2 = int((y + h) / h_img * 511)
                # Add padding
                pad = 10
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(511, x2 + pad)
                y2 = min(511, y2 + pad)
            else:
                x1, y1, x2, y2 = 0, 0, 511, 511

            boxes = torch.tensor(
                [[x1, y1, x2, y2]], dtype=torch.float32, device=self.device
            )

            with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=self.device == "cuda"
            ):
                mask_logits, class_logits = self.model(img_tensor, boxes=boxes)

            probs = torch.softmax(class_logits, dim=1)[0].cpu().numpy()
            pred_class = int(probs.argmax())
            confidence = float(probs.max())
            mask_prob = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()

            # Build all-classes summary
            all_classes = []
            for i in range(len(CLASS_NAMES)):
                all_classes.append(f"{CLASS_NAMES[i]}: {probs[i]:.1%}")
            class_summary = " | ".join(all_classes)

            # Update UI on main thread
            self.root.after(
                0, self._update_result, img_rgb, mask_prob, pred_class, confidence,
                class_summary, f"Box: [{x1},{y1},{x2},{y2}]"
            )

        except Exception as e:
            self.root.after(0, self._on_error, str(e))

    def _update_result(self, img_rgb, mask_prob, pred_class, confidence,
                       class_summary="", box_info=""):
        class_name = (
            CLASS_NAMES[pred_class]
            if pred_class < len(CLASS_NAMES)
            else f"Class {pred_class}"
        )

        if pred_class == 0:
            status_text = (
                f"RESULT: [Good/No Defect] {confidence:.1%}  |  {box_info}"
            )
            color = "green"
        else:
            status_text = (
                f"RESULT: [{class_name}] {confidence:.1%}  |  {box_info}"
            )
            color = "red"

        # Show all class probabilities in the detail label
        self.detail_var.set(class_summary)

        # Overlay — resize mask to match image dimensions
        mask_bin = (mask_prob > 0.5).astype(np.uint8)
        mask_resized = cv2.resize(
            mask_bin, (img_rgb.shape[1], img_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        overlay = img_rgb.copy().astype(np.float32)
        red_overlay = np.zeros_like(overlay)
        red_overlay[:, :, 0] = 255.0
        mask_3d = mask_resized[:, :, None]
        overlay = np.where(mask_3d, overlay * 0.6 + red_overlay * 0.4, overlay)
        overlay = overlay.astype(np.uint8)

        self.ax_right.clear()
        self.ax_right.imshow(overlay)
        self.ax_right.set_title("Prediction Overlay")
        self.ax_right.axis("off")
        self.canvas_right.draw()

        self.status_var.set(status_text)
        self.status_label.config(foreground=color)

        self.is_running = False
        self.run_btn.config(state=tk.NORMAL)

    def _on_error(self, error_msg):
        self.status_var.set(f"ERROR: {error_msg}")
        self.is_running = False
        self.run_btn.config(state=tk.NORMAL)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="IndustrialSAM Desktop App")
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    root = tk.Tk()
    app = IndustrialSAMApp(root, args.checkpoint, device=device)
    root.mainloop()


if __name__ == "__main__":
    main()
