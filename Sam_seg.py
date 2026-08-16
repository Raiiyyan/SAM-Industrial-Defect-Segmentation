import gradio as gr
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_TYPE = "vit_b"
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"  # The base model you downloaded

# Uses your GPU if available, otherwise defaults to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. LOAD MODEL
# ==========================================
print("Loading SAM Base Model...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)

# We initialize the Automatic Mask Generator (no YOLO or prompts required)
mask_generator = SamAutomaticMaskGenerator(sam)
print("Model loaded successfully! Launching UI...")

# ==========================================
# 3. PROCESSING FUNCTION
# ==========================================
def auto_segment(image):
    if image is None:
        return None
        
    print("Scanning image for objects...")
    
    # SAM analyzes the image and returns a list of dictionaries for every object
    masks = mask_generator.generate(image)
    
    if len(masks) == 0:
        print("No objects found.")
        return image
        
    # Sort masks by area (draw large background objects first, small foreground objects on top)
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    
    overlay = image.copy()
    
    for ann in sorted_masks:
        m = ann['segmentation']
        
        # Generate a random RGB color for each distinct object
        color = np.random.randint(0, 255, 3).tolist()
        
        # Blend the colored mask into the original image
        overlay[m] = overlay[m] * 0.4 + np.array(color) * 0.6
        
        # Draw a thin white border around the object to make it pop
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

    print(f"Done! Found {len(masks)} distinct objects.")
    return overlay

# ==========================================
# 4. GRADIO UI
# ==========================================
with gr.Blocks(title="SAM Auto-Scanner") as app:
    gr.Markdown("<h1 style='text-align: center;'>✨ Segment Anything: Auto-Scanner</h1>")
    gr.Markdown("<p style='text-align: center;'>Upload any image. The base model will attempt to find and color every distinct object.</p>")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Drop or Choose Image Here", type="numpy")
            run_btn = gr.Button("🔍 Segment Everything", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Segmented Output", interactive=False)
            
    run_btn.click(fn=auto_segment, inputs=input_image, outputs=output_image)

if __name__ == "__main__":
    app.launch(inbrowser=True, quiet=True)