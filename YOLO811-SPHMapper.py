import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from ultralytics import YOLO
from collections import Counter
from tqdm import tqdm
from scipy.ndimage import center_of_mass
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Spacer
from reportlab.lib.units import inch
 
# Path configuration
input_folder   = r"D:\YOLO-SPHMapper\...\image 444"         # folder with input images
results_folder = r"D:\YOLO-SPHMapper\...\image 444 output"  # output folder
os.makedirs(results_folder, exist_ok=True)
 
# Model loading
seg_model = YOLO(r'...\best_yolo_8_segmentator.pt')   # YOLOv8m-Seg
clf_model = YOLO(r'...\best_yolo_11_classifier.pt')   # YOLOv11-Cls
det_model = YOLO(r'...\best16batch.pt')               # YOLOv8m-Det
 
# Colour maps
bgr_colors    = {"Low": (74, 192, 3), "Moderate": (0, 255, 255), "High": (0, 8, 255)}
hex_colors    = {"Low": "#03C04A",  "Moderate": "#FFFF00",    "High": "#FF0800"}
defect_colors = {"none": "#A0A0A0", "crack": "#0000FF",      "bird": "#A52A2A"}
 
# Adaptive tile size: 32x32 for images < 1000px, 50x50 otherwise
def get_tile_size(img_shape):
    return 32 if max(img_shape) < 1000 else 50
 
# Sort detected panel masks from top-left to bottom-right
def sort_masks_top_left_to_bottom_right(masks_list):
    mask_infos = []
    for mask in masks_list:
        cy, cx = center_of_mass(mask > 0)
        if not np.isnan(cx) and not np.isnan(cy):
            mask_infos.append(((int(cy), int(cx)), mask))
    sorted_masks = sorted(mask_infos, key=lambda x: (x[0][0], x[0][1]))
    return [(i + 1, mask) for i, (_, mask) in enumerate(sorted_masks)]
 
# Collect image filenames and initialise PDF report
panel_names = sorted([f for f in os.listdir(input_folder)
                      if f.lower().endswith((".jpg", ".png"))])
pdf_path = os.path.join(results_folder, "FINAL_RESULTS.pdf")
doc      = SimpleDocTemplate(pdf_path, pagesize=A4)
elements = []
 
# ============================================================
# Main inference loop
# ============================================================
for filename in tqdm(panel_names, desc="Processing"):
    name     = os.path.splitext(filename)[0]
    img_path = os.path.join(input_folder, filename)
    img      = cv2.imread(img_path)
    h, w     = img.shape[:2]
    tile_size = get_tile_size((h, w))
 
    # ----------------------------------------------------------
    # Stage 1: Panel surface segmentation (YOLOv8m-Seg)
    # ----------------------------------------------------------
    seg_result    = seg_model(img)[0]
    masks         = seg_result.masks
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    raw_masks     = []
 
    for m in masks.data:
        m         = m.cpu().numpy().astype(np.uint8) * 255
        m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        combined_mask = np.maximum(combined_mask, m_resized)
        raw_masks.append(m_resized)
 
    panel_masks  = sort_masks_top_left_to_bottom_right(raw_masks)
    masked_img   = cv2.bitwise_and(img, img, mask=combined_mask)
    overlay      = img.copy().astype(np.float32)
    alpha        = 0.5
    panel_stats  = []
    panel_labels = {}
 
    # ----------------------------------------------------------
    # Stage 2: Anomaly detection (YOLOv8m-Det)
    # ----------------------------------------------------------
    det_res   = det_model(img)[0]
    det_boxes = det_res.boxes
    det_map   = {}
 
    for box in det_boxes:
        cls_name = det_res.names[int(box.cls)]
        if cls_name not in ["Bird Drop", "Cracked"]:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        for panel_id, msk in panel_masks:
            if np.any(msk[y1:y2, x1:x2] > 0):
                det_map.setdefault(panel_id, set()).add(cls_name)
 
    # ----------------------------------------------------------
    # Stage 3: Tile-level dust severity classification (YOLOv11-Cls)
    # ----------------------------------------------------------
    for panel_id, msk in panel_masks:
        classes, coords, tile_masks, tile_shapes = [], [], [], []
 
        # Deterministic mask-constrained spatial tiling
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                y_end     = min(y + tile_size, h)
                x_end     = min(x + tile_size, w)
                tile_mask = msk[y:y_end, x:x_end]
                if np.any(tile_mask > 0):          # only tiles within panel mask
                    tile = masked_img[y:y_end, x:x_end]
                    pred = clf_model(tile)[0]
                    cls  = pred.names[int(pred.probs.top1)].capitalize()
                    classes.append(cls)
                    coords.append((x, y))
                    tile_masks.append(tile_mask)
                    tile_shapes.append((x_end - x, y_end - y))
 
        # Apply semi-transparent colour overlay per dust class
        for (x, y), cls, tile_mask, (tile_w, tile_h) in zip(
                coords, classes, tile_masks, tile_shapes):
            if cls not in bgr_colors:
                continue
            color     = np.array(bgr_colors[cls], dtype=np.float32)
            region    = overlay[y:y + tile_h, x:x + tile_w]
            mask_bool = tile_mask > 0
            for c in range(3):
                region[..., c][mask_bool] = (
                    (1 - alpha) * region[..., c][mask_bool]
                    + alpha * color[c])
 
        # Annotate panel centroid with label
        cy, cx = center_of_mass(msk > 0)
        if not np.isnan(cx) and not np.isnan(cy):
            cx, cy     = int(cx), int(cy)
            label_list = ["Dusty"]
            if panel_id in det_map:
                if "Cracked"   in det_map[panel_id]:
                    label_list.append("Cracked")
                if "Bird Drop" in det_map[panel_id]:
                    label_list.append("Bird drop")
            label_text            = f"Panel {panel_id} ({', '.join(label_list)})"
            panel_labels[panel_id] = label_text
            cv2.putText(overlay, label_text, (cx - 80, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
 
        counts = Counter(classes)
        panel_stats.append((panel_labels[panel_id], counts,
                            det_map.get(panel_id, set())))
 
    # Save dust severity overlay image
    overlay_img  = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay_path = os.path.join(results_folder, f"{name}_overlay.png")
    cv2.imwrite(overlay_path, overlay_img)
 
    # ----------------------------------------------------------
    # Dashboard: dual-layer pie-chart per panel
    # ----------------------------------------------------------
    cols = 4
    rows = (len(panel_stats) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = axs.flatten()
 
    for idx, (panel_label, counts, dets) in enumerate(panel_stats):
        ax             = axs[idx]
        labels, sizes, colors = [], [], []
 
        for k in ["Low", "Moderate", "High"]:
            if k in counts:
                labels.append(k)
                sizes.append(counts[k])
                colors.append(hex_colors[k])
 
        # Inner circle: dust severity distribution
        ax.pie(sizes,
               labels=labels if sizes else None,
               colors=colors if sizes else ["#DDDDDD"],
               autopct="%1.1f%%" if sizes else None,
               startangle=90, radius=1.0)
        ax.set_title(panel_label, fontsize=10)
 
        # Outer ring: anomaly type encoding
        outer_colors, outer_sizes = [], []
        if dets:
            if "Cracked"   in dets:
                outer_colors.append(defect_colors["crack"])
                outer_sizes.append(0.5)
            if "Bird Drop" in dets:
                outer_colors.append(defect_colors["bird"])
                outer_sizes.append(0.5)
        if not outer_colors:
            outer_colors = [defect_colors["none"]]
            outer_sizes  = [1]
        ax.pie(outer_sizes, colors=outer_colors, radius=1.25,
               startangle=90,
               wedgeprops=dict(width=0.18, edgecolor='white'))
        ax.axis("equal")
 
    # Hide unused subplots
    for idx in range(len(panel_stats), len(axs)):
        axs[idx].axis("off")
 
    plt.tight_layout()
    chart_path = os.path.join(results_folder, f"{name}_dashboard.png")
    plt.savefig(chart_path)
    plt.close()
 
    # Append dashboard and overlay to PDF report
    elements.append(RLImage(chart_path,   width=6*inch, height=6*inch))
    elements.append(Spacer(1, 20))
    elements.append(RLImage(overlay_path, width=6*inch, height=6*inch))
    elements.append(Spacer(1, 40))
 
# Build and save the final PDF report
doc.build(elements)
print("DONE: FINAL_RESULTS.pdf created.")

