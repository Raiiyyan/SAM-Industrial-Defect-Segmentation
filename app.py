import gradio as gr
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# ============================================================
# 1. CONFIGURATION & MODEL LOADING
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading YOLOv8 Defect Detector...")
YOLO_WEIGHTS = r"G:\CSE465\best.pt"
yolo_model = YOLO(YOLO_WEIGHTS)

print("Loading SAM (Segment Anything Model)...")
SAM_TYPE = "vit_b"
SAM_CHECKPOINT = r"G:\CSE465\sam_vit_b_01ec64.pth"
sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

print("🚀 Pipeline fully loaded and ready for industrial inspection!")

# ============================================================
# 2. HYBRID PIPELINE FUNCTION
# ============================================================
def inspect_product(image, confidence_threshold):
    if image is None:
        return None, "No image provided."

    # Convert RGB (Gradio default) to BGR for OpenCV processing
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Run YOLO detection using the optimal threshold
    results = yolo_model(img_bgr, conf=confidence_threshold, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() # Get bounding boxes
    
    if len(boxes) == 0:
        # Status message for a perfect product
        return image, "✅ STATUS: PASS (No defects detected by YOLO)"

    # Set image for SAM predictor
    sam_predictor.set_image(image)
    
    overlay = image.copy()
    total_defect_pixels = 0

    # For each box found by YOLO, let SAM draw the exact pixel mask
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Format box for SAM input [xmin, ymin, xmax, ymax]
        input_box = np.array([x1, y1, x2, y2])
        
        masks, scores, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False,
        )
        
        if len(masks) > 0:
            mask = masks[0]
            total_defect_pixels += np.sum(mask)
            
            # Apply a vibrant red translucent color over the defect mask
            color = np.array([255, 0, 0], dtype=np.uint8)
            overlay[mask] = overlay[mask] * 0.5 + color * 0.5
            
            # Draw YOLO bounding box outline in bright yellow
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

    status_message = f"❌ STATUS: FAIL (Defect Detected! Total Mask Area: {int(total_defect_pixels)} pixels)"
    return overlay, status_message

# ============================================================
# 3. GRADIO USER INTERFACE
# ============================================================
with gr.Blocks(title="Industrial Hybrid Inspection System") as demo:
    gr.Markdown("<h1 style='text-align: center;'>🏭 Industrial Anomaly Detection & Segmentation</h1>")
    gr.Markdown("<p style='text-align: center;'>Stage 1: YOLOv8 locates the defect. Stage 2: SAM isolates the precise pixel mask.</p>")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Upload Product Test Image", type="numpy")
            conf_slider = gr.Slider(
                minimum=0.1, maximum=0.95, value=0.63, step=0.01, 
                label="YOLO Confidence Threshold (Scientifically Optimized at 0.63)"
            )
            submit_btn = gr.Button("🔍 Run Inspection", variant="primary")
            
        with gr.Column():
            output_img = gr.Image(label="Inspection Result Overlay", interactive=False)
            status_box = gr.Textbox(label="Quality Control Verdict", interactive=False)
            
    submit_btn.click(
        fn=inspect_product, 
        inputs=[input_img, conf_slider], 
        outputs=[output_img, status_box]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, quiet=True)