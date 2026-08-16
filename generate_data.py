import os
import cv2
import shutil
import random

# ============================================================
# 1. CONFIGURATION
# ============================================================
MAIN_DATASET_ROOT = r"G:\CSE465\Datasets"

TRAIN_ROOT = r"G:\CSE465\Dataset_for_training80"
TEST_ROOT = r"G:\CSE465\Dataset_for_testing20"

TRAIN_IMG = os.path.join(TRAIN_ROOT, "images")
TRAIN_LBL = os.path.join(TRAIN_ROOT, "labels")
TEST_IMG  = os.path.join(TEST_ROOT, "images")
TEST_LBL  = os.path.join(TEST_ROOT, "labels")

# Recreate clean target folders
for folder in [TRAIN_IMG, TRAIN_LBL, TEST_IMG, TEST_LBL]:
    os.makedirs(folder, exist_ok=True)

# Set seed for reproducible 80/20 splitting
random.seed(42)

print("🚀 Scanning dataset and splitting 80% Train / 20% Test...")

total_train_imgs = 0
total_test_imgs = 0
total_train_boxes = 0
total_test_boxes = 0

product_folders = [f for f in os.listdir(MAIN_DATASET_ROOT) if os.path.isdir(os.path.join(MAIN_DATASET_ROOT, f))]
print(f"📦 Found {len(product_folders)} products: {', '.join(product_folders)}\n")

# ============================================================
# 2. HELPER TO CONVERT MASKS TO YOLO FORMAT
# ============================================================
def process_defect_item(img_path, mask_path, out_img_dir, out_lbl_dir, prefix):
    global total_train_boxes, total_test_boxes
    
    img_name = os.path.basename(img_path)
    base_name = os.path.splitext(img_name)[0]
    new_img_name = f"{prefix}_{img_name}"
    txt_name = f"{prefix}_{base_name}.txt"
    
    shutil.copy(img_path, os.path.join(out_img_dir, new_img_name))
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0
        
    _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    h, w = mask.shape
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    yolo_lines = []
    box_count = 0
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh < 4:
            continue
            
        x_center = (x + bw / 2.0) / w
        y_center = (y + bh / 2.0) / h
        norm_bw  = bw / w
        norm_bh  = bh / h
        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_bw:.6f} {norm_bh:.6f}")
        box_count += 1
        
    with open(os.path.join(out_lbl_dir, txt_name), "w") as f:
        f.write("\n".join(yolo_lines))
        
    return box_count

# ============================================================
# 3. MASTER LOOP: PROCESS AND SPLIT DATA
# ============================================================
for product in product_folders:
    print(f"--- Processing Product: {product.upper()} ---")
    
    product_dir = os.path.join(MAIN_DATASET_ROOT, product)
    p_test_root = os.path.join(product_dir, "test")
    p_gt_root   = os.path.join(product_dir, "ground_truth")
    
    # --------------------------------------------------------
    # A. PROCESS DEFECTS (80% Train, 20% Test)
    # --------------------------------------------------------
    if os.path.exists(p_test_root):
        subfolders = [f for f in os.listdir(p_test_root) if os.path.isdir(os.path.join(p_test_root, f))]
        defect_categories = [f for f in subfolders if f.lower() != "good"]
        
        for category in defect_categories:
            cat_test_dir = os.path.join(p_test_root, category)
            cat_gt_dir   = os.path.join(p_gt_root, category)
            
            if not os.path.exists(cat_gt_dir):
                continue
                
            img_files = [f for f in os.listdir(cat_test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            random.shuffle(img_files)
            
            split_idx = int(len(img_files) * 0.8)
            train_files = img_files[:split_idx]
            test_files  = img_files[split_idx:]
            
            prefix = f"{product}_{category}"
            
            # Train split (80%)
            for img_name in train_files:
                base_name = os.path.splitext(img_name)[0]
                mask_path = os.path.join(cat_gt_dir, f"{base_name}_mask.png")
                if not os.path.exists(mask_path):
                    alt_path = os.path.join(cat_gt_dir, img_name)
                    if os.path.exists(alt_path): mask_path = alt_path
                    else: continue
                
                boxes = process_defect_item(os.path.join(cat_test_dir, img_name), mask_path, TRAIN_IMG, TRAIN_LBL, prefix)
                total_train_boxes += boxes
                total_train_imgs += 1
                
            # Test split (20%)
            for img_name in test_files:
                base_name = os.path.splitext(img_name)[0]
                mask_path = os.path.join(cat_gt_dir, f"{base_name}_mask.png")
                if not os.path.exists(mask_path):
                    alt_path = os.path.join(cat_gt_dir, img_name)
                    if os.path.exists(alt_path): mask_path = alt_path
                    else: continue
                
                boxes = process_defect_item(os.path.join(cat_test_dir, img_name), mask_path, TEST_IMG, TEST_LBL, prefix)
                total_test_boxes += boxes
                total_test_imgs += 1

    # --------------------------------------------------------
    # B. PROCESS GOOD BACKGROUND SAMPLES (80% Train, 20% Test)
    # --------------------------------------------------------
    good_dir = os.path.join(product_dir, "train", "good")
    if os.path.exists(good_dir):
        good_files = [f for f in os.listdir(good_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(good_files)
        
        split_idx = int(len(good_files) * 0.8)
        train_good = good_files[:split_idx]
        test_good  = good_files[split_idx:]
        
        # Train good samples
        for img_name in train_good:
            base_name = os.path.splitext(img_name)[0]
            new_img_name = f"{product}_good_{img_name}"
            shutil.copy(os.path.join(good_dir, img_name), os.path.join(TRAIN_IMG, new_img_name))
            with open(os.path.join(TRAIN_LBL, f"{product}_good_{base_name}.txt"), "w") as f:
                f.write("")
            total_train_imgs += 1
            
        # Test good samples
        for img_name in test_good:
            base_name = os.path.splitext(img_name)[0]
            new_img_name = f"{product}_good_{img_name}"
            shutil.copy(os.path.join(good_dir, img_name), os.path.join(TEST_IMG, new_img_name))
            with open(os.path.join(TEST_LBL, f"{product}_good_{base_name}.txt"), "w") as f:
                f.write("")
            total_test_imgs += 1

# ============================================================
# 4. FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("✅ Dataset Splitting Complete!")
print(f"📁 Dataset_for_training80: {total_train_imgs} images ({total_train_boxes} defect boxes)")
print(f"📁 Dataset_for_testing20:  {total_test_imgs} images ({total_test_boxes} defect boxes)")
print("="*60)