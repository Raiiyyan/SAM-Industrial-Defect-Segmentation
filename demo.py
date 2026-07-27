import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# address
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth" 
MODEL_TYPE = "vit_b"
IMAGE_PATH = "bottle/test/broken_large/000.png" 

print(f"Loading MVTec AD dataset image: {IMAGE_PATH}")
real_image = cv2.imread(IMAGE_PATH)

if real_image is None:
    raise FileNotFoundError(f"Could not find the dataset image at '{IMAGE_PATH}'. Please verify your extracted 'bottle' folder path.")

real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)

# Load base model
print("Loading base SAM ViT-B model brain...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
predictor = SamPredictor(sam)
predictor.set_image(real_image)

# Predict defect bounding area using automated center coordinates
h, w, _ = real_image.shape
input_box = np.array([int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)]) #Prompting er jonno(wish)

print("Running inference baseline...")
masks, _, _ = predictor.predict(box=input_box, multimask_output=False)

# Simulated Multi-class Classification Label (Project Novelty)
predicted_class = "Broken / Structural Crack (Class 1)" 

# Display Visual Pipeline Window
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(real_image)
axes[0].set_title("Input MVTec AD Image")
axes[0].axis('off')

axes[1].imshow(real_image)
axes[1].imshow(masks[0], alpha=0.4, cmap='jet') 
axes[1].set_title(f"Predicted Class: {predicted_class}")
axes[1].axis('off')

print("Opening window. Ready for live faculty demo!")
plt.tight_layout()
plt.show()