import os
import cv2
import shutil

# ============================================================
# 1. CONFIGURATION
# ============================================================
# Point this to the main folder containing ALL products
MAIN_DATASET_ROOT = r"G:\CSE465\Datasets"

YOLO_ROOT = r"G:\CSE465\YOLO_Dataset"
IMG_DIR = os.path.join(YOLO_ROOT, "images", "train")
LBL_DIR = os.path.join(YOLO_ROOT, "labels", "train")

# Clean/recreate target folders
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

print("🚀 Scanning main dataset directory for all products...")

total_images = 0
total_boxes = 0

# Get all product folders (bottle, cable, capsule, etc.)
product_folders = [f for f in os.listdir(MAIN_DATASET_ROOT) if os.path.isdir(os.path.join(MAIN_DATASET_ROOT, f))]
print(f"📦 Found {len(product_folders)} product categories: {', '.join(product_folders)}\n")

# ============================================================
# 2. MASTER LOOP: PROCESS EVERY PRODUCT
# ============================================================
for product in product_folders:
    print(f"--- Processing Product: {product.upper()} ---")
    
    product_dir = os.path.join(MAIN_DATASET_ROOT, product)
    TEST_ROOT = os.path.join(product_dir, "test")
    GT_ROOT   = os.path.join(product_dir, "ground_truth")
    
    # --------------------------------------------------------
    # A. PROCESS DEFECT CATEGORIES (POSITIVE SAMPLES)
    # --------------------------------------------------------
    if os.path.exists(TEST_ROOT):
        subfolders = [f for f in os.listdir(TEST_ROOT) if os.path.isdir(os.path.join(TEST_ROOT, f))]
        defect_categories = [f for f in subfolders if f.lower() != "good"]
        
        for category in defect_categories:
            cat_test_dir = os.path.join(TEST_ROOT, category)
            cat_gt_dir   = os.path.join(GT_ROOT, category)
            
            if not os.path.exists(cat_gt_dir):
                continue
                
            for img_name in os.listdir(cat_test_dir):
                if not img_name.endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                base_name = os.path.splitext(img_name)[0]
                mask_name = f"{base_name}_mask.png"
                
                img_path  = os.path.join(cat_test_dir, img_name)
                mask_path = os.path.join(cat_gt_dir, mask_name)
                
                if not os.path.exists(mask_path):
                    alt_path = os.path.join(cat_gt_dir, img_name)
                    if os.path.exists(alt_path): mask_path = alt_path
                    else: continue

                # Prefix the product name to avoid file overwriting
                new_img_name = f"{product}_{category}_{img_name}"
                shutil.copy(img_path, os.path.join(IMG_DIR, new_img_name))
                
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None: continue
                _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
                h, w = mask.shape
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                yolo_lines = []
                for cnt in contours:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    if bw * bh < 4: continue  # Skip microscopic noise
                    
                    x_center = (x + bw / 2.0) / w
                    y_center = (y + bh / 2.0) / h
                    norm_bw  = bw / w
                    norm_bh  = bh / h
                    yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_bw:.6f} {norm_bh:.6f}")
                    total_boxes += 1
                
                txt_name = f"{product}_{category}_{base_name}.txt"
                with open(os.path.join(LBL_DIR, txt_name), "w") as f:
                    f.write("\n".join(yolo_lines))
                
                total_images += 1

    # --------------------------------------------------------
    # B. PROCESS NORMAL IMAGES (NEGATIVE / BACKGROUND SAMPLES)
    # --------------------------------------------------------
    good_dir = os.path.join(product_dir, "train", "good")
    if os.path.exists(good_dir):
        good_count = 0
        for img_name in os.listdir(good_dir):
            if not img_name.endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(good_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            
            new_img_name = f"{product}_good_{img_name}"
            shutil.copy(img_path, os.path.join(IMG_DIR, new_img_name))
            
            # Create an EMPTY text file
            txt_name = f"{product}_good_{base_name}.txt"
            with open(os.path.join(LBL_DIR, txt_name), "w") as f:
                f.write("")
                
            good_count += 1
            total_images += 1
            
        print(f"  -> Added {good_count} normal 'good' images.")

# ============================================================
# 3. FINAL SUMMARY
# ============================================================
print("\n" + "="*50)
print(f"[SUCCESS] Massive Universal Dataset Assembled!")
print(f"Total images processed: {total_images}")
print(f"Total defect bounding boxes generated: {total_boxes}")
print(f"Your multi-product YOLO dataset is ready at: {YOLO_ROOT}")
print("="*50)