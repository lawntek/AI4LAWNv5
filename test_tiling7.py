import os
import sys

pdal_dll_path = r"C:\Users\user\miniconda3\envs\lidar_env\Library\bin"
os.environ['PATH'] = pdal_dll_path + os.pathsep + os.environ['PATH']
import pdal
import os
import time
import uuid
import json
import math
import argparse
import requests
from collections import namedtuple
from urllib.parse import unquote
import numpy as np
import cv2
# import laspy
import scipy.ndimage as nd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from shapely.geometry import Polygon
from skimage.io import imread, imsave
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from pyproj import Transformer, CRS
from cv2geojson import find_geocontours
from extendBoundaryMapBox import extendBoundary
from model import Segmenter_segformer

os.chdir(os.path.dirname(os.path.abspath(__file__)))

TILE_SIZE = 768
TILE_OVERLAP = 128
BATCH_SIZE = 4
INPUT_SHAPE = 768

# MEAN = [0.485, 0.456, 0.406, 0.5, 0.5]
# STD = [0.229, 0.224, 0.225, 0.225, 0.225]
CH_3_NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
CH_5_NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5, 0.5],
                         std=[0.229, 0.224, 0.225, 0.225, 0.225])
])
VIZ_COLORMAP = {
    0: (255, 100, 0),
    1: (61, 245, 61),
    2: (0, 0, 0),
}

app = Flask(__name__)
CORS(app)
os.makedirs("images", exist_ok=True)
os.makedirs("masks", exist_ok=True)

def opt():
    parser = argparse.ArgumentParser(description='DeepLawn Fast-Tiled 5-Channel Server')
    parser.add_argument('--model_path', type=str, default='./models/lawn_model.pth')
    parser.add_argument('--tree_model_path', type=str, default='./models/tree_model.pth')
    parser.add_argument('--metadata_path', type=str, default=None)
    parser.add_argument('--mode', type=str, choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--tile_size', type=int, default=TILE_SIZE)
    parser.add_argument('--tile_overlap', type=int, default=TILE_OVERLAP)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--model_name', type=str, default='mit_b5')
    return parser.parse_args()

def modify_model_for_5_channels(model):
    """
    Finds and modifies the first patch embedding layer of a SegFormer-based UNet encoder.
    """
    try:
        # Based on your Sub-module Discovery: unet.encoder.patch_embed1.proj
        target_layer = model.unet.encoder.patch_embed1.proj
        print(f"✅ Found target layer: {target_layer}")
    except AttributeError as e:
        print("--- Sub-module Discovery ---")
        # Fallback to help find the layer if the path changes
        for name, module in model.named_modules():
            if "patch_embed1.proj" in name:
                print(f"Potential layer found at: {name}")
        raise AttributeError(f"Could not find encoder patch_embed layer: {e}")

    # 1. Extract original weights [out_channels, 3, K, K]
    old_weights = target_layer.weight.data 
    
    # 2. Create new conv layer with 5 input channels
    new_conv = nn.Conv2d(
        in_channels=5, 
        out_channels=target_layer.out_channels,
        kernel_size=target_layer.kernel_size,
        stride=target_layer.stride,
        padding=target_layer.padding,
        bias=(target_layer.bias is not None)
    ).to(old_weights.device)
    
    # 3. Initialize weights
    with torch.no_grad():
        # Copy original RGB weights
        new_conv.weight[:, :3, :, :] = old_weights
        # Initialize LiDAR channels (4 & 5) with mean of RGB weights
        new_conv.weight[:, 3:, :, :] = old_weights.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)
        if target_layer.bias is not None:
            new_conv.bias = target_layer.bias
            
    # 4. Replace the layer back into the encoder
    model.unet.encoder.patch_embed1.proj = new_conv
    
    print("🚀 Successfully modified SegFormer-UNet encoder for 5-channel input.")
    return model
import os, requests, uuid, cv2, torch
import numpy as np
import scipy.ndimage as nd
from pyproj import Transformer, CRS
from affine import Affine

import json
import numpy as np
import scipy.ndimage as nd

import requests
import json
import pdal
import numpy as np
import cv2
from pyproj import Transformer
from affine import Affine
from scipy import ndimage as nd

USGS_EPT_INDEX = "https://usgs-lidar-public.s3.amazonaws.com/index/ept.json"

def load_usgs_ept_index():
    try:
        r = requests.get(USGS_EPT_INDEX, timeout=20)
        r.raise_for_status()
        print("✅ Loaded USGS EPT catalog")
        return r.json()
    except Exception as e:
        print(f"❌ Failed to load index: {e}")
        return {"datasets": []}


import requests
import json
from urllib.parse import quote

def find_ept_url(lat, lon):
    """
    Queries USGS National Map 3DEPElevationIndex to find a public EPT URL
    for the given lat/lon (WGS84). Returns the ept.json URL or None.
    """
    base_url = "https://index.nationalmap.gov/arcgis/rest/services/3DEPElevationIndex/MapServer/"
    
    # Try more layers — ordered roughly most recent/reliable first
    layers = [24, 25, 23, 22, 7, 6, 0, 1, 8, 9]  

    geometry = f'{{"x":{lon},"y":{lat},"spatialReference":{{"wkid":4326}}}}'

    params = {
        "geometry": geometry,
        "geometryType": "esriGeometryPoint",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "false",
        "f": "json"
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }

    found_layer = None
    best_url = None
    candidates = []

    for layer_id in layers:
        try:
            url = f"{base_url}{layer_id}/query"
            r = requests.get(url, params=params, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            features = data.get("features", [])

            if features:
                found_layer = layer_id
                attr = features[0]["attributes"]
                
                print(f"✓ Found match in layer {layer_id} (first feature)")
                print("Full attributes:", json.dumps(attr, indent=2))

                # 1. Direct EPT link in any field (most reliable)
                for key, value in attr.items():
                    if isinstance(value, str) and "ept.json" in value.lower():
                        print(f"🎯 Direct EPT link found in '{key}': {value}")
                        if value.startswith("http"):
                            return value
                        # Sometimes relative — try to fix
                        if value.startswith("/"):
                            return "https://index.nationalmap.gov" + value

                # 2. Collect all possible name/project/workunit fields
                possible_names = []
                name_keys = [
                    "workunit", "project", "name", "Project", "LidarProjectName",
                    "CollectionName", "lpc_name", "project_name", "dataset_name",
                    "workunit_id", "project_id", "collect_name"
                ]
                for k in name_keys:
                    v = attr.get(k)
                    if v and isinstance(v, str) and len(v.strip()) > 3:
                        possible_names.append(v.strip())

                # Also check link/metadata fields that might contain project hint
                for k, v in attr.items():
                    if isinstance(v, str) and ("project" in k.lower() or "name" in k.lower() or "workunit" in k.lower()):
                        if v not in possible_names:
                            possible_names.append(v)

                print(f"Possible name/project strings: {possible_names}")

                # 3. Build and test candidates
                base_bucket = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/"

                for raw_name in possible_names:
                    # Clean aggressively but keep meaningful parts
                    clean = raw_name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
                    clean = clean.replace(",", "").replace("/", "_").replace("__", "_").strip("_")

                    patterns = [
                        clean,
                        f"USGS_LPC_{clean}",
                        f"USGS_LPC_{clean}_LAS_20{clean[-2:]}" if clean[-2:].isdigit() else f"USGS_LPC_{clean}",
                        f"USGS_LPC_{clean}_LAS_2019",  # common for 2017 collections processed later
                        f"USGS_LPC_{clean}_B17",
                        f"USGS_LPC_{clean}_B19",
                        clean.replace("SEWRPC", "Southeast_WI"),
                        clean.replace("SEWRPC", "SE_WI"),
                        f"WI_{clean}"
                    ]

                    for pat in patterns:
                        candidate = f"{base_bucket}{quote(pat)}/ept.json"
                        if candidate not in candidates:
                            candidates.append(candidate)

                # Test candidates (HEAD is fast)
                for cand in candidates:
                    print(f"Testing → {cand}")
                    try:
                        h = requests.head(cand, timeout=6, allow_redirects=True)
                        if h.status_code in (200, 403):  # 403 can still mean exists (S3 policy)
                            print(f"VALID EPT FOUND: {cand}")
                            best_url = cand
                            break  # take first working one
                        elif h.status_code == 404:
                            pass  # silent for clean output
                        else:
                            print(f"  → status {h.status_code}")
                    except Exception as e:
                        print(f"  → error: {e.__class__.__name__}")

                if best_url:
                    return best_url

                print(f"No valid EPT found after testing {len(candidates)} candidates in layer {layer_id}")

        except Exception as e:
            print(f"Layer {layer_id} failed: {e.__class__.__name__} - {str(e)}")

    if candidates and not best_url:
        print("\nMost promising candidate (pick manually if needed):")
        print(candidates[0])  # often the USGS_LPC_... one

    print(f"❌ No EPT URL discovered across {len(layers)} layers")
    return None







def get_lidar_derivatives(metadata, width, height):
    try:
        ept_url = find_ept_url(metadata["center_lat"], metadata["center_lng"])
    except:
        print("❌ No MetaData found")
        return np.zeros((height, width, 2), np.float32)

    if not ept_url:
        print("❌ No EPT found")
        return np.zeros((height, width, 2), np.float32)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    w, s = transformer.transform(metadata["bound_west"], metadata["bound_south"])
    e, n = transformer.transform(metadata["bound_east"], metadata["bound_north"])

    pipeline = {
        "pipeline": [
            {
                "type": "readers.ept",
                "filename": ept_url,
                "bounds": f"([{w},{e}],[{s},{n}])",
                "resolution": 1.0   # 👈 controls point density (IMPORTANT)
            },
            {"type": "filters.range", "limits": "Classification![7:7]"}
        ]
    }

    pipe = pdal.Pipeline(json.dumps(pipeline))
    pipe.execute()

    if not pipe.arrays:
        return np.zeros((height, width, 2), np.float32)

    pts = pipe.arrays[0]
    affine = Affine((e-w)/width, 0, w, 0, -(n-s)/height, n)

    intensity = pts["Intensity"].astype(np.float32)
    z = pts["Z"]

    intensity_grid = burn_points_to_grid(pts["X"], pts["Y"], intensity, affine, width, height)
    intensity_grid = nd.grey_dilation(intensity_grid, size=(3,3))  # ← add this! (import scipy.ndimage as nd)

    deforest_grid = burn_points_to_grid(pts["X"], pts["Y"], intensity, affine, width, height)  # intensity, not Z!
    deforest_grid = nd.grey_dilation(deforest_grid, size=(3,3))   # smooth like training
    deforest_grid = cv2.normalize(deforest_grid, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32) / 255.0

    # SAVE AS GeoTIFFs
    crs = "EPSG:3857"  # Web Mercator — same as your transformed bounds

    # Intensity TIFF
    intensity_meta = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': 'float32',
        'crs': crs,
        'transform': affine,
        'nodata': 0.0,          # or np.nan if you prefer
    }

    intensity_path = "lidar_intensity.tif"
    with rasterio.open(intensity_path, 'w', **intensity_meta) as dst:
        dst.write(intensity_grid, 1)
    print(f"Saved intensity grid → {intensity_path}")

    # Elevation TIFF
    elev_meta = intensity_meta.copy()  # same metadata, single band
    elev_path = "lidar_ground_elevation.tif"
    with rasterio.open(elev_path, 'w', **elev_meta) as dst:
        dst.write(deforest_grid, 1)
    print(f"Saved ground elevation grid → {elev_path}")
    return np.stack([intensity_grid, deforest_grid], axis=-1)

def burn_points_to_grid(x, y, vals, trans, w, h):
    # Precise training burn logic: round + int cast
    cols, rows = ~trans * (x, y)
    cols, rows = np.round(cols).astype(int), np.round(rows).astype(int)
    mask = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    grid = np.zeros((h, w), dtype=np.float32)
    if np.any(mask):
        grid[rows[mask], cols[mask]] = vals[mask]
    return grid

@torch.no_grad()
def infer_big_image(model, img_pil, lidar_np, num_classes, device,norm_transform,
                   tile=768, overlap=128, batch_size=6, is_5ch=True):
    model.eval()
    img_np = np.array(img_pil)
    H, W, _ = img_np.shape


    if H <= tile and W <= tile:
            print(f"Image ({W}x{H}) is smaller than tile size. Performing single pass.")
            
            # Resize RGB and LiDAR to the model's required INPUT_SHAPE (768)
            img_resized = cv2.resize(img_np, (INPUT_SHAPE, INPUT_SHAPE), interpolation=cv2.INTER_LINEAR)
            lidar_resized = cv2.resize(lidar_np, (INPUT_SHAPE, INPUT_SHAPE), interpolation=cv2.INTER_LINEAR)

            if is_5ch:
                # Stack to (768, 768, 5)
                full_stack = np.concatenate([img_resized, lidar_resized], axis=2).astype(np.float32)
                mean = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5]).to(device).view(-1, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225, 0.225, 0.225]).to(device).view(-1, 1, 1)
                n_channels = 5
            else:
                full_stack = img_resized.astype(np.float32)
                mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(-1, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(-1, 1, 1)
                n_channels = 3

            # Convert to Tensor (CHW) and Normalize
            img_tensor = torch.from_numpy(full_stack).permute(2, 0, 1).to(device)
            img_tensor = (img_tensor / 255.0 - mean) / std
            img_tensor = img_tensor.unsqueeze(0) # Add batch dimension

            # Model Inference
            out = model(img_tensor)
            if isinstance(out, (list, tuple)): out = out[0]
            
            # Resize output back to original H, W
            out_resized = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
            return torch.argmax(out_resized, dim=0).cpu().numpy().astype(np.uint8)

    # ---------------------------------------------------------
    # CONDITIONAL STACKING: Only stack if the model expects 5ch
    # ---------------------------------------------------------
    if is_5ch:
        full_stack = np.concatenate([img_np, lidar_np], axis=2).astype(np.float32)
        print("full_stack: "+str(full_stack.shape))
        n_channels = 5
    else:
        full_stack = img_np.astype(np.float32)
        n_channels = 3


    

    stride = tile - overlap
    logits_accum = torch.zeros((num_classes, H, W), device=device)
    weight_accum = torch.zeros((H, W), device=device)
    hann1d = torch.hann_window(tile, periodic=False, device=device)
    w2d = torch.outer(hann1d, hann1d).clamp(min=1e-6)
    
    coords = []
    for y in range(0, H, stride):
        y1 = min(y + tile, H); y0 = max(0, y1 - tile)
        for x in range(0, W, stride):
            x1 = min(x + tile, W); x0 = max(0, x1 - tile)
            coords.append((y0, y1, x0, x1))

    batch_imgs, batch_coords = [], []
    for (y0, y1, x0, x1) in coords:
        crop = full_stack[y0:y1, x0:x1, :]
        if crop.shape[0] != tile or crop.shape[1] != tile:
            crop = cv2.resize(crop, (tile, tile), interpolation=cv2.INTER_LINEAR)
        
        # --- FIX STARTS HERE ---
        # Instead of Image.fromarray(crop), convert directly to Tensor
        # HWC (Height, Width, Channels) -> CHW (Channels, Height, Width)
        img_tensor = torch.from_numpy(crop).permute(2, 0, 1).float()
        
        # Apply normalization manually if needed, or extract it from your norm_transform
        # Since transforms.Normalize usually expects a tensor, we do this:
        if is_5ch:
            # Manually apply CH_5_NORM's normalization part
            mean = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5]).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225, 0.225, 0.225]).view(-1, 1, 1)
            img_tensor = (img_tensor / 255.0 - mean) / std
        else:
            # Manually apply CH_3_NORM's normalization part
            mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            img_tensor = (img_tensor / 255.0 - mean) / std

        batch_imgs.append(img_tensor)
        # --- FIX ENDS HERE ---

        batch_coords.append((y0, y1, x0, x1))


        if len(batch_imgs) == batch_size:
            _process_batch(model, batch_imgs, batch_coords, logits_accum, weight_accum, w2d, device, tile)
            batch_imgs.clear(); batch_coords.clear()

    if batch_imgs:
        _process_batch(model, batch_imgs, batch_coords, logits_accum, weight_accum, w2d, device, tile)

    logits_accum /= weight_accum
    return torch.argmax(logits_accum, dim=0).cpu().numpy().astype(np.uint8)

def _process_batch(model, batch_imgs, batch_coords, logits_accum, weight_accum, w2d, device, tile):
    batch = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
        out = model(batch)
        if isinstance(out, (list, tuple)): out = out[0]
    for i, (y0, y1, x0, x1) in enumerate(batch_coords):
        h_crop, w_crop = y1 - y0, x1 - x0
        logits_tile = F.interpolate(out[i].unsqueeze(0), size=(tile, tile), mode='bilinear', align_corners=False).squeeze(0)
        logits_accum[:, y0:y1, x0:x1] += logits_tile[:, :h_crop, :w_crop] * w2d[:h_crop, :w_crop]
        weight_accum[y0:y1, x0:x1] += w2d[:h_crop, :w_crop]

def class2color(mask, labels):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label in labels: color_mask[mask == label.trainId] = label.color
    return color_mask

def apply_colormap_index(mask_idx):
    h, w = mask_idx.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, col in VIZ_COLORMAP.items(): out[mask_idx == k] = col
    return out

def overlay_on_image(img_pil, mask_rgb, alpha=0.3):
    img_np = np.array(img_pil).astype(np.float32)
    m_np = mask_rgb.astype(np.float32)
    blend_mask = np.any(mask_rgb != [0,0,0], axis=-1)
    blended = img_np.copy()
    blended[blend_mask] = (1 - alpha) * img_np[blend_mask] + alpha * m_np[blend_mask]
    return blended.astype(np.uint8)

def remove_overlapping_polygons(polygons, overlap_threshold=0.8):
    if not polygons: return []
    unique_polygons = []
    polygons_to_remove = set()
    for i, poly1 in enumerate(polygons):
        if i in polygons_to_remove: continue
        try:
            poly1_shapely = Polygon(poly1['geometry']['coordinates'][0])
            if not poly1_shapely.is_valid: continue
        except: continue
        for j, poly2 in enumerate(polygons):
            if i == j or j in polygons_to_remove: continue
            try:
                poly2_shapely = Polygon(poly2['geometry']['coordinates'][0])
                if not poly2_shapely.is_valid: continue
                if poly1_shapely.intersects(poly2_shapely):
                    intersection_area = poly1_shapely.intersection(poly2_shapely).area
                    min_area = min(poly1_shapely.area, poly2_shapely.area)
                    if min_area > 0 and (intersection_area / min_area) > overlap_threshold:
                        if poly1_shapely.area < poly2_shapely.area:
                            polygons_to_remove.add(i)
                            break
                        else: polygons_to_remove.add(j)
            except: continue
    for i, poly in enumerate(polygons):
        if i not in polygons_to_remove: unique_polygons.append(poly)
    return unique_polygons

def test_lawn(base_dir, model, image_path, metadata, device, tile=TILE_SIZE, overlap=TILE_OVERLAP, batch=BATCH_SIZE):
    Label = namedtuple('label', ['name', 'color', 'trainId'])
    labels = [Label('background', [0, 0, 0], 0), Label('driveway', [245, 147, 49], 2), Label('sidewalk', [61, 61, 245], 3), Label('road', [255, 255, 255], 1), Label('turf', [61, 245, 61], 5), Label('building', [255, 106, 77], 4)]
    COLORS = {lbl.trainId: np.array(lbl.color, dtype=np.uint8) for lbl in labels}
    start = time.time()
    img = Image.open(image_path).convert('RGB')
    W, H = img.size
    lidar_extra = get_lidar_derivatives(metadata, W, H)
    print(np.unique(lidar_extra))
    pred_idx = infer_big_image(model, img, lidar_extra, 6, device, CH_5_NORM,
                               tile, overlap, batch, is_5ch=True)
    color_out = class2color(pred_idx, labels)
    ts = int(time.time() * 1000)
    uid = uuid.uuid4().hex[:8]
    os.makedirs(base_dir, exist_ok=True)
    img.save(f'./images/image_{ts}_{uid}.png')
    class_names = {2: "driveway", 3: "sidewalk", 4: "building", 5: "turf", 1: "road"}
    for class_id, name in class_names.items():
        mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        mask_rgb[pred_idx == class_id] = COLORS[class_id]
        overlay_path = os.path.join(base_dir, f"{name}_overlay.png")
        Image.fromarray(overlay_on_image(img, mask_rgb, 0.5)).save(overlay_path)
    print(f"[STUDENT MODEL] Inference on {W}x{H} done in {time.time() - start:.2f}s")
    return color_out, pred_idx

def test_tree(base_dir, model, image_path, device, tile=TILE_SIZE, overlap=TILE_OVERLAP, batch=BATCH_SIZE):
    start = time.time()
    img = Image.open(image_path).convert('RGB')
    W, H = img.size
    
    # Passing None/Dummy for LiDAR because is_5ch=False will ignore it
    dummy_lidar = np.zeros((H, W, 2), dtype=np.float32)
    
    # FIX: Pass is_5ch=False here
    pred_idx = infer_big_image(model, img, dummy_lidar, 2, 
                               device,CH_3_NORM, tile=tile, overlap=overlap, 
                               batch_size=batch, is_5ch=False)
    
    rgb_mask = np.zeros((H, W, 3), dtype=np.uint8)
    rgb_mask[pred_idx == 1] = [255, 0, 0] 
    out_overlay = os.path.join(base_dir, "tree_overlay.png")
    os.makedirs(base_dir, exist_ok=True)
    Image.fromarray(overlay_on_image(img, rgb_mask, 0.4)).save(out_overlay)
    print(f"[TREE] Inference on {W}x{H} image done in {time.time() - start:.2f}s")
    return rgb_mask

def rgb_to_class(mask):
    class_map = np.zeros(mask.shape[:2], dtype=np.uint8)
    for (r, g, b), idx in {(61, 245, 61): 1, (255, 100, 0): 2, (245, 147, 49): 3, (61, 61, 245): 4, (255, 0, 0): 5, (0, 0, 0): 0}.items():
        class_map[(mask == (r, g, b)).all(axis=2)] = idx
    return class_map

def mask_to_geojson(mask, class_id, color):
    binary = (mask == class_id).astype(np.uint8)
    geoc = find_geocontours(binary, mode='opencv')
    features = [c.export_feature(color=color, label='roi') for c in geoc]
    return json.dumps(remove_overlapping_polygons(features))

def draw_geojson_polygons(shape, geojson_data, color):
    H, W = shape
    mask = np.zeros((H, W, 3), dtype=np.uint8)
    if isinstance(geojson_data, str):
        try: geo = json.loads(geojson_data)
        except: return mask
    else: geo = geojson_data
    features = []
    if isinstance(geo, dict) and geo.get("type") == "FeatureCollection": features = geo.get("features", [])
    elif isinstance(geo, dict) and geo.get("type") == "Feature": features = [geo]
    elif isinstance(geo, dict) and geo.get("type") in ("Polygon", "MultiPolygon"): features = [{"geometry": geo}]
    elif isinstance(geo, list):
        for item in geo:
            if isinstance(item, dict) and "geometry" in item: features.append(item)
            elif isinstance(item, dict) and item.get("type") in ("Polygon", "MultiPolygon"): features.append({"geometry": item})
    for f in features:
        geom = f.get("geometry")
        if geom is None: continue
        gtype = geom.get("type")
        if gtype == "Polygon":
            rings = geom.get("coordinates", [])
            if rings: cv2.fillPoly(mask, [np.array(rings[0], dtype=np.int32)], color)
        elif gtype == "MultiPolygon":
            for poly in geom.get("coordinates", []):
                if poly: cv2.fillPoly(mask, [np.array(poly[0], dtype=np.int32)], color)
    return mask

def geojson_to_mask_robust(geojson_input, height, width, fill_value=255):
    shapes = []
    if isinstance(geojson_input, str):
        try: data = json.loads(geojson_input)
        except: return np.zeros((height, width), dtype=np.uint8)
    else: data = geojson_input
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                geom = item.get("geometry") or item
                if geom: shapes.append((geom, fill_value))
            else: shapes.append((item, fill_value))
    elif isinstance(data, dict):
        if data.get("type") == "FeatureCollection":
            for feat in data.get("features", []):
                if feat.get("geometry"): shapes.append((feat.get("geometry"), fill_value))
        elif data.get("type") == "Feature":
            if data.get("geometry"): shapes.append((data.get("geometry"), fill_value))
        else: shapes.append((data, fill_value))
    if not shapes: return np.zeros((height, width), dtype=np.uint8)
    return rasterize(shapes=shapes, out_shape=(height, width), transform=Affine.translation(0, 0)*Affine.scale(1, 1), fill=0, dtype=np.uint8, all_touched=True, default_value=fill_value)

def ensure_binary_mask(mask_rgb):
    if mask_rgb.ndim == 3: mask_gray = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
    else: mask_gray = mask_rgb
    _, mask_bin = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
    return mask_bin.astype(np.uint8)

def get_metadata_from_csv(csv_path, target_filename):
    """
    Look up the specific row in the user-provided CSV that matches the current image.
    Columns: [filename, center_lat, center_lng, ..., bound_north, bound_south, bound_east, bound_west, ...]
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        # Match by filename (handling potential path differences)
        base_name = os.path.basename(target_filename)
        row = df[df['filename'].str.contains(base_name, na=False)]
        
        if row.empty:
            print(f"⚠️ No metadata row found for {base_name} in {csv_path}")
            return None
        
        # Extract the specific columns you listed
        res = row.iloc[0].to_dict()
        metadata_dict = {
            'bound_north': float(res['bound_north']),
            'bound_south': float(res['bound_south']),
            'bound_east': float(res['bound_east']),
            'bound_west': float(res['bound_west'])
        }
        return metadata_dict
    except Exception as e:
        print(f"❌ Error reading metadata CSV: {e}")
        return None
# -------------------------------------------------
# Flask endpoint (FULLY FIXED)
# -------------------------------------------------
@app.route('/model/predict-boundary', methods=['GET', 'POST'])
def predict_endpoint():
    # ---------- 1. Parse request ----------
    try:
        if request.method == 'GET':
            boundary_pts_raw = request.args.get('boundaryPoints', '[]')
            features_raw = request.args.get('features', '[]')
            boundary_pts = json.loads(unquote(boundary_pts_raw))
            features = json.loads(unquote(features_raw))
# --- SMART METADATA DISCOVERY ---
            metadata_dict = None
            raw_metadata = request.form.get('metadata')
            
            if raw_metadata:
                # Case A: Metadata sent as a JSON string field
                metadata_dict = json.loads(raw_metadata)
            else:
                # Case B: Metadata sent as individual form fields (Filename, bound_north, etc.)
                metadata_dict = request.form.to_dict()
            map_url = request.args.get('mapurl')
            uploaded_file = None
        else:
            boundary_pts_raw = request.form.get('boundaryPoints', '[]')
            features_raw = request.form.get('features', '[]')
            boundary_pts = json.loads(boundary_pts_raw)
            features = json.loads(features_raw)
            uploaded_file = request.files.get('map_file')
            metadata_dict = None
            raw_metadata = request.form.get('metadata')
            
            if raw_metadata:
                # Case A: Metadata sent as a JSON string field
                metadata_dict = json.loads(raw_metadata)
            else:
                # Case B: Metadata sent as individual form fields (Filename, bound_north, etc.)
                metadata_dict = request.form.to_dict()
            map_url = None
    except Exception as e:
        return jsonify({"error": f"JSON parse error: {e}"}), 400


    # ---------- 3. Load image ----------
    ts = int(time.time() * 1000)
    uid = uuid.uuid4().hex[:8]
    session_id = f"{ts}_{uid}"
    base_dir = f"./predict_boundary/{session_id}"
    os.makedirs(base_dir, exist_ok=True)
    img_path = f"./images/{ts}_{uid}_map_image.jpg"

    try:
        if uploaded_file:
            uploaded_file.save(img_path)
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        elif map_url:
            image = imread(map_url)
        else:
            return jsonify({"error": "No image source provided"}), 400

        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        imsave(img_path, image)
        img_pil = Image.fromarray(image).convert('RGB')
        W, H = img_pil.size

    except Exception as e:
        if os.path.exists(img_path):
            os.remove(img_path)
        return jsonify({"error": f"Image load error: {e}"}), 400

    # ---------- 4. Run student & tree models ----------
    try:
        student_mask_image = None
        tree_mask_image = None

        # Student model (driveway, sidewalk, building, turf, road)
        if any(x in features for x in ['LAWN','BUILDING','DRIVEWAY','SIDEWALK','ROAD']):
            # This call uses our metadata_dict (which is either a valid dict or None)
            # Student model call
            color_out, pred_idx = test_lawn(
                base_dir=base_dir,
                model=GLOBAL_STUDENT_MODEL,
                image_path=img_path,
                metadata=metadata_dict,
                device=GLOBAL_DEVICE,
                tile=GLOBAL_TILE,
                overlap=GLOBAL_OVERLAP,
                batch=GLOBAL_BATCH
            )
            student_mask_image = pred_idx

        # Tree model (usually remains 3-channel unless retrained)
        if 'TREE' in features:
            tree_mask_image = test_tree(
                base_dir=base_dir,
                model=GLOBAL_TREE_MODEL,
                image_path=img_path,
                device=GLOBAL_DEVICE,
                tile=GLOBAL_TILE,
                overlap=GLOBAL_OVERLAP,
                batch=GLOBAL_BATCH
            )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Inference error: {e}"}), 500

    # ---------- 5. Priority Merge & GeoJSON Extraction ----------
    prediction_data = {}
    try:
        priority = student_mask_image.copy() if student_mask_image is not None else None

        # Mapping for GeoJSON extraction
        mapping = {"DRIVEWAY": 2, "SIDEWALK": 3, "BUILDING": 4, "LAWN": 5, "ROAD": 1}
        colors = {"DRIVEWAY": (245, 147, 49), "SIDEWALK": (61, 61, 245), 
                  "BUILDING": (255, 106, 77), "LAWN": (61, 245, 61), "ROAD": (255, 255, 255)}

        if priority is not None:
            for feat in features:
                if feat in mapping:
                    prediction_data[feat] = mask_to_geojson(priority, mapping[feat], colors[feat])

        if 'TREE' in features and tree_mask_image is not None:
            # Convert tree RGB mask back to binary for GeoJSON
            tree_binary = np.all(tree_mask_image == (255, 0, 0), axis=-1).astype(np.uint8)
            prediction_data["TREE"] = mask_to_geojson(tree_binary, 1, (255, 0, 0))

    except Exception as e:
        return jsonify({"error": f"GeoJSON processing error: {e}"}), 500
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

    return jsonify(prediction_data)

@app.route('/model/extend-boundary/<boundary>', methods=['GET'])
def extend_boundary_endpoint(boundary):
    try: return jsonify(extendBoundary(json.loads(unquote(boundary))))
    except: return jsonify(boundary)

if __name__ == '__main__':
    args = opt()
    device = torch.device('cuda' if torch.cuda.is_available() and args.mode == 'gpu' else 'cpu')
    GLOBAL_STUDENT_MODEL = Segmenter_segformer(args.model_name, classes=6).to(device)
    GLOBAL_STUDENT_MODEL = modify_model_for_5_channels(GLOBAL_STUDENT_MODEL)
    GLOBAL_STUDENT_MODEL.load_state_dict(torch.load(args.model_path, map_location=device))
    GLOBAL_STUDENT_MODEL.eval()
    GLOBAL_TREE_MODEL = Segmenter_segformer(args.model_name, classes=2).to(device)
    GLOBAL_TREE_MODEL.load_state_dict(torch.load(args.tree_model_path, map_location=device))
    GLOBAL_TREE_MODEL.eval()
    GLOBAL_DEVICE, GLOBAL_TILE, GLOBAL_OVERLAP, GLOBAL_BATCH = device, args.tile_size, args.tile_overlap, args.batch_size
    # COLOR_CLASS_MAP = {(245,147,49): 1, (61,61,245): 2, (255,106,77): 3, (61,245,61): 4, (255,255,255): 5, (255,0,0): 6}
    app.run(host='0.0.0.0', debug=False, port=5000, threaded=True)