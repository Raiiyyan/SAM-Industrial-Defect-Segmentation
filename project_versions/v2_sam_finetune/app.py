"""
IndustrialSAM v2 — Defect detection
    python app.py
"""

import os
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAM_WEIGHTS = os.path.join(SCRIPT_DIR, "sam_vit_b_01ec64.pth")
CKPT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "best_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_VOCAB = {
    0: "Flawless", 1: "Surface Scratch", 2: "Structural Crack",
    3: "Hole / Puncture", 4: "Inclusion", 5: "Missing Component",
    6: "Discoloration / Stain", 7: "Geometric Deformation",
}

_model = None


def get_model():
    global _model
    if _model is None:
        from model import IndustrialSAM
        print("Loading model...")
        _model = IndustrialSAM(SAM_WEIGHTS, device=DEVICE, train_resolution=512)
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        _model.load_state_dict(ckpt["model_state_dict"], strict=False)
        _model.eval()
        print("Model ready.")
    return _model


def predict(img_np):
    if img_np is None:
        raise gr.Error("Upload an image first.")

    h, w = img_np.shape[:2]
    rgb = img_np[:, :, :3].copy()

    resized = np.array(Image.fromarray(rgb).resize((512, 512), Image.BILINEAR))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = ((tensor - mean) / std).unsqueeze(0).to(DEVICE)

    model = get_model()
    with torch.no_grad():
        mask_logits, class_logits = model(tensor)

    probs = torch.softmax(class_logits, dim=1)[0]
    pred_id = probs.argmax().item()
    conf = probs[pred_id].item()

    mask_prob = torch.sigmoid(mask_logits[0, 0]).cpu().numpy()
    mask_resized = np.array(
        Image.fromarray((mask_prob * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    base = rgb.astype(np.float32) / 255.0
    red = np.array([1.0, 0.0, 0.0])
    alpha = np.clip(mask_resized[:, :, None] * 2.0, 0, 0.7)
    overlay = (1.0 - alpha) * base + alpha * red
    overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    top5 = torch.topk(probs, min(5, probs.shape[0]))
    lines = []
    for val, idx in zip(top5.values, top5.indices):
        i = idx.item()
        bar_len = int(val.item() * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        marker = " ◀" if i == pred_id else ""
        lines.append(f"{CLASS_VOCAB[i]:22s} {val.item():5.1%}  {bar}{marker}")
    ranking = "\n".join(lines)

    icon = "⚠️ DEFECT" if pred_id != 0 else "✅ FLAWLESS"
    label = f"{icon}: {CLASS_VOCAB[pred_id]}  ({conf:.1%})"

    return overlay, label, ranking


with gr.Blocks(
    title="IndustrialSAM v2",
    theme=gr.themes.Soft(),
    css=".infer-btn { height: 50px !important; font-size: 18px !important; }"
) as demo:
    gr.Markdown(
        "<h1 style='text-align:center'>🔍 Industrial Defect Segmentation</h1>"
        "<p style='text-align:center;color:#888'>Upload an image — one click to detect defects</p>"
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(label="Upload Image", type="numpy")
            run_btn = gr.Button("🔍  Detect Defect", variant="primary", elem_classes="infer-btn")

        with gr.Column(scale=1):
            img_out = gr.Image(label="Result", type="numpy")
            label_out = gr.Textbox(label="Prediction", interactive=False)
            rank_out = gr.Textbox(label="All Classes", lines=6, interactive=False)

    run_btn.click(predict, inputs=[img_in], outputs=[img_out, label_out, rank_out])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
