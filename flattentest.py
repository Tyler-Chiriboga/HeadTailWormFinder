import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image

# YOLOv7 local imports (ensure you have a local clone with models/ and utils/)
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

# Segment Anything
from segment_anything import sam_model_registry, SamPredictor

# CNN (ResNet18) 
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models

# For skeletonization + morphology
from skimage.morphology import skeletonize
from collections import deque

###################################################
# 0. Helper: get_device()
###################################################
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################################################
# 1. YOLOv7 Loader
###################################################
def load_yolov7_model(weights_path, device, conf_thres=0.25, iou_thres=0.45, img_size=640):
    model = attempt_load(weights_path, map_location=device)
    model.eval().to(device)

    model.conf = conf_thres
    model.iou = iou_thres
    model.img_size = img_size

    def detect_image(bgr_img):
        letterboxed, _, _ = letterbox(bgr_img, new_shape=model.img_size, auto=False)
        letterboxed = letterboxed[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB
        letterboxed = np.ascontiguousarray(letterboxed)

        im = torch.from_numpy(letterboxed).float().to(device)
        im /= 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        with torch.no_grad():
            pred = model(im)[0]

        det = non_max_suppression(pred, model.conf, model.iou, classes=None, agnostic=False)
        return det

    model.detect_image = detect_image
    return model

###################################################
# 2. SAM Loader
###################################################
def load_sam_predictor(checkpoint_path, model_type="vit_h", device=None):
    if device is None:
        device = get_device()
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    return SamPredictor(sam)

###################################################
# 3. CNN Model Loader
###################################################
def load_health_model(weights_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from scipy.interpolate import splprep, splev
from collections import deque

########################################
# Helper Functions for Mask Smoothing
########################################
def remove_small_regions(mask, min_size=50):
    """
    Removes connected components smaller than 'min_size' in a binary mask.
    mask: np.uint8 => 0 or 1
    """
    # Convert to bool
    mask_bool = (mask > 0)
    # Remove tiny objects
    cleaned_bool = remove_small_objects(mask_bool, min_size=min_size)
    # Convert back to 0/1
    return cleaned_bool.astype(np.uint8)

def smooth_mask_morphology(mask, open_ksize=3, close_ksize=3):
    """
    Applies morphological open->close to reduce jagged edges.
    mask: np.uint8 => 0 or 1
    Returns: smoothed mask (binary 0/1)
    """
    if mask.max() == 1:
        # Convert 0/1 => 0/255 for OpenCV
        mask = (mask * 255).astype(np.uint8)

    # Structuring elements
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))

    # Morphological open -> remove small protrusions
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    # Morphological close -> fill small gaps
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close)

    # Convert back to 0/1
    return (mask_closed > 127).astype(np.uint8)

def fill_holes(mask):
    """
    Fill internal holes in a binary mask using scipy.ndimage's binary_fill_holes.
    mask: np.uint8 => 0 or 1
    """
    filled = binary_fill_holes(mask > 0)
    return filled.astype(np.uint8)

def smooth_mask_blur(mask, ksize=5, sigma=1):
    """
    Smooth a binary mask by Gaussian blur + threshold.
    mask: np.uint8 (0 or 1)
    Returns: smoothed mask (binary)
    """
    if mask.max() == 1:
        mask = (mask * 255).astype(np.uint8)

    mask_float = mask.astype(np.float32)
    mask_blurred = cv2.GaussianBlur(mask_float, (ksize, ksize), sigma)
    # Re-threshold at half of 255 => ~127
    _, mask_thresholded = cv2.threshold(mask_blurred, 127, 255, cv2.THRESH_BINARY)
    return (mask_thresholded > 127).astype(np.uint8)

########################################
# Optional: Smooth Normal Vectors
########################################
def smooth_normals(norm_unit, window=5):
    """
    Simple moving-average smoothing of normal vectors to reduce jags.
    norm_unit: (n_points, 2)
    window: how many points to average over
    """
    if window < 2:
        return norm_unit  # no smoothing needed

    # pad replicate on both ends
    pad_size = window // 2
    padded = np.pad(norm_unit, ((pad_size, pad_size), (0,0)), mode='edge')

    smoothed = np.zeros_like(norm_unit)

    for i in range(norm_unit.shape[0]):
        # slice local window
        start = i
        end = i + window
        local_vecs = padded[start:end]
        mean_vec = np.mean(local_vecs, axis=0)
        mag = np.linalg.norm(mean_vec)
        if mag > 1e-8:
            smoothed[i] = mean_vec / mag
        else:
            smoothed[i] = mean_vec

    return smoothed

########################################
# Main Straightening Function
########################################
def straighten_worm_preserve_girth(
    mask_crop,
    worm_crop,
    spacing=10.0,
    spline_smooth=10.0,
    normal_smooth_window=5,
    min_object_size=50,
    open_ksize=3,
    close_ksize=3,
    blur_ksize=0
):
    """
    Straightens a segmented worm while preserving body thickness.

    Steps:
      1) (Optional) Clean the mask:
         - fill holes
         - remove small objects
         - morphological open/close
         - optional blur+threshold
      2) Skeletonize
      3) BFS from endpoint to endpoint -> centerline
      4) Spline interpolation
      5) Distance transform => local half-width
      6) (Optional) Smooth normal vectors
      7) Sample into unwrapped image

    Args:
      mask_crop (np.uint8): binary worm mask, shape (H, W), 0/1
      worm_crop (np.uint8): color image (H, W, 3) BGR or RGB
      spacing (float): arc-length spacing for the unwrapped columns
      spline_smooth (float): 's' parameter in splprep (higher => more smoothing)
      normal_smooth_window (int): size for normal vector smoothing
      min_object_size (int): remove small connected regions below this size
      open_ksize, close_ksize (int): morphological open/close kernel sizes
      blur_ksize (int): if >0, apply a final cv2.GaussianBlur to the straightened output

    Returns:
      straightened (np.uint8) or None if something fails
    """

    H, W = mask_crop.shape[:2]
    if worm_crop.shape[0] != H or worm_crop.shape[1] != W:
        print("Mask and worm_crop dimensions do not match.")
        return None

    # (A) Clean / Smooth the mask
    # Fill holes
    mask_filled = fill_holes(mask_crop)
    # Remove small objects
    mask_removed = remove_small_regions(mask_filled, min_object_size)
    # Morphological open/close
    mask_morph = smooth_mask_morphology(mask_removed, open_ksize, close_ksize)

    # We can optionally skip or add blur. Example:
    # mask_blurred = smooth_mask_blur(mask_morph, ksize=3, sigma=1)
    # For now, we'll just keep the morphological version
    cleaned_mask = mask_morph

    # (B) Skeletonize
    skel = skeletonize(cleaned_mask > 0)
    skel_pts = np.argwhere(skel)
    if len(skel_pts) < 2:
        print("Skeleton is too small.")
        return None

    # (C) Find endpoints for BFS
    def neighbors(r, c):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if (dr == 0 and dc == 0):
                    continue
                rr, cc = r+dr, c+dc
                if 0 <= rr < H and 0 <= cc < W:
                    if skel[rr, cc]:
                        yield (rr, cc)

    endpoint_list = []
    for r, c in skel_pts:
        deg = sum(1 for _ in neighbors(r, c))
        if deg == 1:
            endpoint_list.append((r, c))

    if len(endpoint_list) < 2:
        print("No valid endpoints found; can't BFS.")
        return None

    # If more than 2 endpoints, pick the pair that are farthest apart
    if len(endpoint_list) > 2:
        max_dist = 0
        best_pair = (endpoint_list[0], endpoint_list[1])
        for i in range(len(endpoint_list)):
            for j in range(i+1, len(endpoint_list)):
                e1 = endpoint_list[i]
                e2 = endpoint_list[j]
                dist = np.hypot(e1[0]-e2[0], e1[1]-e2[1])
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (e1, e2)
        start, goal = best_pair
    else:
        # Just take first and last in the list
        start = endpoint_list[0]
        goal  = endpoint_list[-1]

    visited = set([start])
    parent = {}
    queue = deque([start])
    found = False
    while queue and not found:
        node = queue.popleft()
        if node == goal:
            found = True
            break
        for nbr in neighbors(*node):
            if nbr not in visited:
                visited.add(nbr)
                parent[nbr] = node
                queue.append(nbr)

    if not found:
        print("BFS did not find a path from start to goal.")
        return None

    # Reconstruct BFS path
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    path.reverse()  # now start->goal

    if len(path) < 10:
        print("Skeleton path is too short.")
        return None

    path = np.array(path)  # shape (N,2) => (row, col)

    # (D) Spline Interpolation
    # compute arc lengths
    dists = np.cumsum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
    dists = np.insert(dists, 0, 0.0)
    total_length = dists[-1]
    if total_length < 1:
        print("Zero-length skeleton path.")
        return None

    try:
        tck, _ = splprep([path[:,0], path[:,1]], u=dists, s=spline_smooth)
    except Exception as e:
        print("splprep failed:", e)
        return None

    n_points = int(max(2, total_length // spacing))
    if n_points < 2:
        print("n_points < 2; spacing might be too large or worm is too short.")
        return None

    s_vals = np.linspace(0, total_length, n_points)
    centerline = np.array(splev(s_vals, tck)).T  # shape (n_points, 2), (row, col)

    # (E) Distance transform => local half-width
    dist_map = distance_transform_edt(cleaned_mask)

    # Derivative => normal
    deriv = np.array(splev(s_vals, tck, der=1)).T  # shape (n_points, 2)
    normals = np.stack([-deriv[:,1], deriv[:,0]], axis=-1)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_unit = normals / (norm_len + 1e-8)

    # (F) Optional normal smoothing
    norm_unit = smooth_normals(norm_unit, window=normal_smooth_window)

    local_halfwidths = []
    for (sx, sy) in centerline:
        r_i = int(round(sx))
        c_i = int(round(sy))
        if 0 <= r_i < H and 0 <= c_i < W:
            hw = dist_map[r_i, c_i]
            local_halfwidths.append(hw)
        else:
            local_halfwidths.append(0.0)

    max_hw = int(np.ceil(max(local_halfwidths)))
    if max_hw < 1:
        print("max_hw < 1; worm too thin or mask is invalid.")
        return None

    # (G) Create output array, white background
    out_h = 2*max_hw + 1
    out_w = n_points
    straightened = np.full((out_h, out_w, 3), 255, dtype=np.uint8)

    # (H) Sample each column
    for i in range(n_points):
        cx, cy = centerline[i]  # row, col
        hw = local_halfwidths[i]
        half_wid = int(np.ceil(hw))

        nx, ny = norm_unit[i]
        center_row = max_hw

        for offset in range(-half_wid, half_wid+1):
            out_r = center_row + offset
            if out_r < 0 or out_r >= out_h:
                continue

            px = cx + offset*nx
            py = cy + offset*ny
            px_i = int(round(px))
            py_i = int(round(py))

            if (0 <= px_i < H) and (0 <= py_i < W):
                if cleaned_mask[px_i, py_i] == 1:
                    straightened[out_r, i] = worm_crop[px_i, py_i]

    # (I) Optional final blur to hide slight pixel jags
    if blur_ksize > 1:
        straightened = cv2.GaussianBlur(straightened, (blur_ksize, blur_ksize), 0)

    return straightened

import cv2
import numpy as np
from skimage.morphology import skeletonize
from collections import deque
from scipy.interpolate import splprep, splev
from scipy.ndimage import distance_transform_edt, map_coordinates

def straighten_worm_subpixel(mask_crop, worm_crop, spacing=10.0):
    """
    Straighten a worm using subpixel interpolation (map_coordinates).
    1) Skeleton + BFS to get centerline
    2) Spline interpolate centerline => (n_points) columns
    3) Determine global max half-width => rows = 2*max_hw + 1
    4) For each (row, col) in unwrapped space, compute (px, py) => map_coordinates

    Args:
        mask_crop:  binary (H,W) mask (0 or 1) for the worm
        worm_crop:  color (H,W,3) image (BGR or RGB)
        spacing:    how many pixels of arc length between centerline samples
    Returns:
        unwrapped:  (out_h, out_w, 3) image, white background
                    or None if something fails
    """

    # 0) Basic checks
    H, W = mask_crop.shape[:2]
    if worm_crop.shape[0] != H or worm_crop.shape[1] != W:
        print("Mask/worm dimensions mismatch.")
        return None

    # 1) Skeleton
    skel = skeletonize(mask_crop > 0)
    skel_pts = np.argwhere(skel)
    if len(skel_pts) < 2:
        print("Skeleton is too small.")
        return None

    # 2) Find endpoints for BFS
    def neighbors(r, c):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r+dr, c+dc
                if 0 <= rr < H and 0 <= cc < W:
                    if skel[rr, cc]:
                        yield (rr, cc)

    endpoint_list = []
    for r, c in skel_pts:
        deg = sum(1 for _ in neighbors(r, c))
        if deg == 1:
            endpoint_list.append((r, c))

    if len(endpoint_list) < 2:
        print("No valid endpoints found.")
        return None

    # Just pick first and last endpoint for BFS
    start = endpoint_list[0]
    goal = endpoint_list[-1]

    # BFS
    visited = set([start])
    parent = {}
    queue = deque([start])
    found = False
    while queue and not found:
        node = queue.popleft()
        if node == goal:
            found = True
            break
        for nbr in neighbors(*node):
            if nbr not in visited:
                visited.add(nbr)
                parent[nbr] = node
                queue.append(nbr)

    if not found:
        print("BFS failed to connect endpoints.")
        return None

    # Reconstruct BFS path
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    path.reverse()  # start -> goal

    if len(path) < 10:
        print("Skeleton path too short.")
        return None

    path = np.array(path)  # shape (N,2) => (row,col)

    # 3) Spline interpolate the centerline with some smoothing
    #    If you want more smoothing, increase 's'
    dists = np.cumsum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
    dists = np.insert(dists, 0, 0.0)
    total_length = dists[-1]
    if total_length < 1:
        print("Zero-length worm.")
        return None

    try:
        tck, _ = splprep([path[:,0], path[:,1]], u=dists, s=5)
    except Exception as e:
        print("splprep error:", e)
        return None

    n_points = int(max(2, total_length // spacing))
    if n_points < 2:
        print("Not enough points; spacing too large or worm too short.")
        return None

    s_vals = np.linspace(0, total_length, n_points)
    centerline = np.array(splev(s_vals, tck)).T  # shape (n_points, 2) => (row, col)

    # 4) Distance transform => local half-width
    dist_map = distance_transform_edt(mask_crop)

    #   Derivative => normal vector
    deriv = np.array(splev(s_vals, tck, der=1)).T  # shape (n_points,2)
    normals = np.stack([-deriv[:,1], deriv[:,0]], axis=-1)
    mag = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_unit = normals / (mag + 1e-8)
    norm_unit = smooth_normals(norm_unit, window=60)

    local_halfwidths = []
    for (sx, sy) in centerline:
        r_i = int(round(sx))
        c_i = int(round(sy))
        if 0 <= r_i < H and 0 <= c_i < W:
            hw = dist_map[r_i, c_i]
        else:
            hw = 0
        local_halfwidths.append(hw)

    local_halfwidths = np.array(local_halfwidths)
    max_hw = int(np.ceil(local_halfwidths.max()))
    if max_hw < 1:
        print("max_hw < 1, worm might be too thin.")
        return None

    # 5) Build a 2D coordinate map for subpixel sampling
    out_h = 2*max_hw + 1
    out_w = n_points
    X_map = np.zeros((out_h, out_w), dtype=np.float32)
    Y_map = np.zeros((out_h, out_w), dtype=np.float32)

    # We define r_vals from -max_hw..+max_hw
    r_vals = np.linspace(-max_hw, max_hw, out_h)

    for i in range(out_w):
        cx, cy = centerline[i]  # row, col
        nx, ny = norm_unit[i]   # normal
        # local half-width
        # (We keep a global max_hw for the output shape,
        #  but you could clamp r if you only want [-local_hw, local_hw].)

        for j in range(out_h):
            r = r_vals[j]
            # compute px, py in the original image
            px = cx + r*nx
            py = cy + r*ny
            X_map[j, i] = px
            Y_map[j, i] = py

    # 6) Use map_coordinates to sample each channel
    worm_R = worm_crop[:,:,0].astype(np.float32)
    worm_G = worm_crop[:,:,1].astype(np.float32)
    worm_B = worm_crop[:,:,2].astype(np.float32)

    R_sample = map_coordinates(worm_R, [X_map, Y_map], order=1, mode='constant', cval=255)
    G_sample = map_coordinates(worm_G, [X_map, Y_map], order=1, mode='constant', cval=255)
    B_sample = map_coordinates(worm_B, [X_map, Y_map], order=1, mode='constant', cval=255)

    unwrapped = np.stack([R_sample, G_sample, B_sample], axis=-1)
    unwrapped = np.clip(unwrapped, 0, 255).astype(np.uint8)

    # 7) Also map the mask to see which pixels are inside/outside worm
    mask_float = mask_crop.astype(np.float32)
    mask_sample = map_coordinates(mask_float, [X_map, Y_map], order=0, mode='constant', cval=0)
    # mask_sample < 0.5 => outside => set to white
    outside = (mask_sample < 0.5)
    unwrapped[outside] = [255, 255, 255]

    return unwrapped



###################################################
# 6. run_inference (with flatten + straighten)
###################################################
def run_inference(
    input_dir,
    yolo_model,
    sam_predictor,
    health_model,
    transform,
    output_dir="annotated_results",
    flattened_dir="flattened_worms",
    straightened_dir="straightened_worms",
    csv_path="results.csv",
    export_mode="both"  # "project", "chip", or "both"
):
    """
    Processes images and .avi in `input_dir`.
     - For .avi, grabs 1st frame
     - YOLO detection -> SAM -> largest contour
     - CNN classification
     - Flatten worm on white background
     - Attempt naive "straightening" using a skeleton BFS approach
     - Writes CSV row for each detection
     
    Args:
        export_mode: "project" - single Excel/CSV for entire project
                     "chip" - separate Excel/CSV per chip
                     "both" - both project-level and per-chip exports
    """
    device = get_device()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(flattened_dir, exist_ok=True)
    os.makedirs(straightened_dir, exist_ok=True)

    results = []
    all_files = sorted(os.listdir(input_dir))
    for file_name in all_files:
        file_path = os.path.join(input_dir, file_name)
        ext = file_name.lower().split('.')[-1]
        if ext not in ['avi', 'png', 'jpg', 'jpeg']:
            continue

        # read .avi or image
        if ext == 'avi':
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                print(f"[WARN] Could not read first frame from {file_path}")
                continue
            original_img = frame
            annotated_basename = f"annotated_{os.path.splitext(file_name)[0]}.png"
        else:
            original_img = cv2.imread(file_path)
            if original_img is None:
                print(f"[WARN] Could not read image {file_path}")
                continue
            annotated_basename = f"annotated_{file_name}"

        h0, w0 = original_img.shape[:2]
        print(f"[INFO] Processing {file_name}, shape={w0}x{h0}, ext={ext}")

        # 1) YOLO detect
        det = yolo_model.detect_image(original_img)
        det = det[0]  # Nx6
        if det is None or len(det) == 0:
            print("   -> No detections found.")
            out_path = os.path.join(output_dir, annotated_basename)
            cv2.imwrite(out_path, original_img)
            continue

        # rescale to original
        new_shape = yolo_model.img_size
        letterboxed, ratio, (dw, dh) = letterbox(original_img, new_shape=new_shape, auto=False)
        im_shape = letterboxed.shape
        det[:, :4] = scale_coords(
            (im_shape[0], im_shape[1]),
            det[:, :4],
            (h0, w0)
        ).round()

        # 2) For each detection => SAM => largest contour => classification
        sam_predictor.set_image(original_img)
        for idx, (*xyxy, conf, cls) in enumerate(det):
            x1, y1, x2, y2 = map(int, xyxy)

            box_np = np.array([[x1, y1, x2, y2]])
            masks, _, _ = sam_predictor.predict(
                point_coords=None,
                box=box_np,
                multimask_output=False
            )
            mask = masks[0].astype(np.uint8)

            # largest contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            xc, yc, bw, bh = cv2.boundingRect(largest_contour)

            # crop worm region
            worm_crop = original_img[yc:yc+bh, xc:xc+bw]
            mask_crop = mask[yc:yc+bh, xc:xc+bw]

            # 3) CNN classification
            worm_crop_rgb = cv2.cvtColor(worm_crop, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(worm_crop_rgb)
            inp_tensor = transform(pil_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                score = health_model(inp_tensor).item()

            if score >= 0.5:
                classification = "Leaky"
                color = (0, 0, 255)
            else:
                classification = "Healthy"
                color = (0, 255, 0)

            # 4) Draw only contour
            #cv2.drawContours(original_img, [largest_contour], -1, color, thickness=2)
            cv2.putText(original_img, f"{classification}:{score:.2f}",
                        (xc, max(0,yc-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 5) Flatten worm => white background
            flatten_name = f"{os.path.splitext(file_name)[0]}_det{idx}_flat.png"
            flatten_path = os.path.join(flattened_dir, flatten_name)
            worm_white_bg = np.full((bh, bw, 3), 255, dtype=np.uint8)
            worm_white_bg[mask_crop == 1] = worm_crop[mask_crop == 1]
            cv2.imwrite(flatten_path, worm_white_bg)

            # 6) Straighten worm => naive skeleton approach
            # straightened_img = straighten_worm_preserve_girth(
            #     mask_crop,
            #     worm_crop,
            #     spacing=1,           # smaller => more columns => smoother unwrapping
            #     spline_smooth=.1,     # bigger => more smoothing of centerline
            #     normal_smooth_window=5,
            #     min_object_size=50,   # remove tiny noise
            #     open_ksize=3,
            #     close_ksize=3,
            #     blur_ksize=3          # final blur kernel for the unwrapped image
            # )
            straightened_img = straighten_worm_subpixel(mask_crop,worm_crop,spacing=1)
            if straightened_img is not None:
                straighten_name = f"{os.path.splitext(file_name)[0]}_det{idx}_straight.png"
                straighten_path = os.path.join(straightened_dir, straighten_name)
                cv2.imwrite(straighten_path, straightened_img)
            else:
                straighten_name = ""

            # store CSV row
            results.append({
                "filename": file_name,
                "score": round(score, 4),
                "classification": classification,
                "xc": xc,
                "yc": yc,
                "bw": bw,
                "bh": bh,
                "contour_area": contour_area,
                "flattened": flatten_name,
                "straightened": straighten_name
            })

        # save annotated
        out_path = os.path.join(output_dir, annotated_basename)
        cv2.imwrite(out_path, original_img)
        print(f"   -> Saved annotated: {out_path}")

    # 7) CSV with summaries based on export_mode
    if len(results) > 0:
        df = pd.DataFrame(results)
        
        # Extract chip name from filename (e.g., "chip1_001.avi" -> "chip1")
        df['chip'] = df['filename'].apply(lambda x: x.rsplit('_', 1)[0] if '_' in x else x.rsplit('.', 1)[0])
        
        # Get base path for exports
        base_path = os.path.dirname(csv_path)
        os.makedirs(base_path, exist_ok=True)
        
        # === Helper function to create chip summary ===
        def create_chip_summary(chip_df, chip_name):
            total = len(chip_df)
            healthy = (chip_df['classification'] == 'Healthy').sum()
            leaky = (chip_df['classification'] == 'Leaky').sum()
            return pd.DataFrame([{
                'metric': 'Chip Name',
                'value': chip_name
            }, {
                'metric': 'Total Worms',
                'value': total
            }, {
                'metric': 'Healthy Count',
                'value': healthy
            }, {
                'metric': 'Leaky Count',
                'value': leaky
            }, {
                'metric': 'Healthy %',
                'value': round(healthy / total * 100, 2) if total > 0 else 0
            }, {
                'metric': 'Leaky %',
                'value': round(leaky / total * 100, 2) if total > 0 else 0
            }, {
                'metric': 'Avg Score',
                'value': round(chip_df['score'].mean(), 4)
            }, {
                'metric': 'Min Score',
                'value': round(chip_df['score'].min(), 4)
            }, {
                'metric': 'Max Score',
                'value': round(chip_df['score'].max(), 4)
            }])
        
        # === Export per chip ===
        if export_mode in ["chip", "both"]:
            chips_dir = os.path.join(base_path, "per_chip_results")
            os.makedirs(chips_dir, exist_ok=True)
            
            for chip_name in df['chip'].unique():
                chip_df = df[df['chip'] == chip_name].copy()
                chip_summary = create_chip_summary(chip_df, chip_name)
                
                # Excel per chip
                chip_excel = os.path.join(chips_dir, f"{chip_name}_results.xlsx")
                with pd.ExcelWriter(chip_excel, engine='openpyxl') as writer:
                    chip_summary.to_excel(writer, sheet_name='Summary', index=False)
                    chip_df.to_excel(writer, sheet_name='Individual Results', index=False)
                
                # CSV per chip
                chip_df.to_csv(os.path.join(chips_dir, f"{chip_name}_results.csv"), index=False)
                chip_summary.to_csv(os.path.join(chips_dir, f"{chip_name}_summary.csv"), index=False)
                
            print(f"[INFO] Saved per-chip exports to {chips_dir}")
        
        # === Export project-level ===
        if export_mode in ["project", "both"]:
            # Per-Chip Summary table
            chip_summary_table = df.groupby('chip').agg(
                total_worms=('filename', 'count'),
                healthy_count=('classification', lambda x: (x == 'Healthy').sum()),
                leaky_count=('classification', lambda x: (x == 'Leaky').sum()),
                avg_score=('score', 'mean'),
                min_score=('score', 'min'),
                max_score=('score', 'max')
            ).reset_index()
            chip_summary_table['healthy_pct'] = (chip_summary_table['healthy_count'] / chip_summary_table['total_worms'] * 100).round(2)
            chip_summary_table['leaky_pct'] = (chip_summary_table['leaky_count'] / chip_summary_table['total_worms'] * 100).round(2)
            
            # Project Summary
            total_worms = len(df)
            healthy_count = (df['classification'] == 'Healthy').sum()
            leaky_count = (df['classification'] == 'Leaky').sum()
            project_summary = pd.DataFrame([{
                'metric': 'Total Worms',
                'value': total_worms
            }, {
                'metric': 'Healthy Count',
                'value': healthy_count
            }, {
                'metric': 'Leaky Count',
                'value': leaky_count
            }, {
                'metric': 'Healthy %',
                'value': round(healthy_count / total_worms * 100, 2) if total_worms > 0 else 0
            }, {
                'metric': 'Leaky %',
                'value': round(leaky_count / total_worms * 100, 2) if total_worms > 0 else 0
            }, {
                'metric': 'Avg Score',
                'value': round(df['score'].mean(), 4)
            }, {
                'metric': 'Min Score',
                'value': round(df['score'].min(), 4)
            }, {
                'metric': 'Max Score',
                'value': round(df['score'].max(), 4)
            }, {
                'metric': 'Total Chips',
                'value': df['chip'].nunique()
            }])
            
            # Save project Excel with multiple sheets
            excel_path = csv_path.replace('.csv', '.xlsx')
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                project_summary.to_excel(writer, sheet_name='Project Summary', index=False)
                chip_summary_table.to_excel(writer, sheet_name='Chip Summary', index=False)
                df.to_excel(writer, sheet_name='Individual Results', index=False)
            print(f"[INFO] Saved project Excel to {excel_path}")
            
            # Save project CSV files
            df.to_csv(csv_path, index=False)
            chip_summary_table.to_csv(csv_path.replace('.csv', '_chip_summary.csv'), index=False)
            project_summary.to_csv(csv_path.replace('.csv', '_project_summary.csv'), index=False)
            print(f"[INFO] Saved project CSV files")
            
            # Print summary to console
            print("\n" + "="*50)
            print("PROJECT SUMMARY")
            print("="*50)
            print(f"Total Worms: {total_worms}")
            print(f"Healthy: {healthy_count} ({healthy_count/total_worms*100:.1f}%)")
            print(f"Leaky: {leaky_count} ({leaky_count/total_worms*100:.1f}%)")
            print(f"Avg Score: {df['score'].mean():.4f}")
            print("\n" + "="*50)
            print("CHIP SUMMARY")
            print("="*50)
            print(chip_summary_table.to_string(index=False))
            print("="*50 + "\n")
        
        print(f"[INFO] Export mode: {export_mode} - completed with {len(results)} worm detections")
    else:
        print("[INFO] No detections; CSV not created.")

###################################################
# 7. Main
###################################################
if __name__ == "__main__":
    """
    This script:
      - YOLOv7 detection + SAM segmentation
      - CNN classification (healthy/leaky)
      - Flatten worm onto white background
      - Attempt naive skeleton BFS => "straightened" worm
      - CSV logs each detection
    """
    device = get_device()
    # data_dir = "/media/share/IX70 data/Leaky Gut Projects/CCAR-001 LKY-1A/segmented_data/"               # Root directory containing "leaky" and "healthy" subfolders
    # output_model_path = "LeakyGutNewModel/worm_leaky_model.pth"        # Where to save our trained ResNet model
    # sam_checkpoint_path = "sam_vit_h_4b8939.pth"  # SAM checkpoint
    # yolov7_weights = "/home/hedtpc/Downloads/YOLO314Full/yolov7-custom_lambda/yolov7-custom/best_Leaky_v3.pt"               # YOLOv7 weights file
    # images_for_inference = "/media/share/IX70 data/DatasetData/LeakyGut//" 
    # inference_output_dir = "/media/share/IX70 data/Leaky Gut Projects/CCAR-001 LKY-1B/inference_results/"  # Where annotated images go
    # example_image_for_gradcam = "example_image.png"
    # Adjust paths:
    yolo_weights = "/home/hedtpc/Downloads/YOLO314Full/yolov7-custom_lambda/yolov7-custom/best_Leaky_v3.pt"
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    health_cnn_weights = "LeakyGutNewModel/worm_leaky_model.pth"
    input_dir = "/media/hedtpc/BulkStorage/LeakyGut/PCSL-001 LHK-1A (reprocess me)/output/"      # images or .avi videos
    output_dir = input_dir+"out_results/"
    flattened_dir = input_dir+"out_results/flattened_worms/"
    straightened_dir = input_dir+"out_results/straightened_worms/"
    csv_path = input_dir+"out_results/results.csv"
    
    # Export mode: "project" (single file), "chip" (per-chip files), or "both"
    export_mode = "both"

    # 1) Load YOLOv7
    yolo_model = load_yolov7_model(
        weights_path=yolo_weights,
        device=device,
        conf_thres=0.25,
        iou_thres=0.45,
        img_size=640
    )

    # 2) Load SAM
    sam_predictor = load_sam_predictor(
        checkpoint_path=sam_checkpoint,
        model_type="vit_h",
        device=device
    )

    # 3) Load Health CNN
    health_model = load_health_model(health_cnn_weights, device)

    # 4) Transform pipeline
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    # 5) Run
    run_inference(
        input_dir=input_dir,
        yolo_model=yolo_model,
        sam_predictor=sam_predictor,
        health_model=health_model,
        transform=transform,
        output_dir=output_dir,
        flattened_dir=flattened_dir,
        straightened_dir=straightened_dir,
        csv_path=csv_path,
        export_mode=export_mode  # "project", "chip", or "both"
    )

    print("[INFO] Done.")
