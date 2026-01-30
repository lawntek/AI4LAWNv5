# --- Standard Libraries ---
import os
import sys
import time
import uuid
import json
import math
import random
import datetime
import warnings
from glob import glob
from io import BytesIO
from collections import namedtuple
from itertools import chain
from urllib.parse import unquote
import argparse

# --- Numerical / Array / Image Processing ---
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# --- PyTorch + Torchvision ---
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# --- Albumentations for image transforms ---
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- skimage ---
from skimage.io import imread, imsave, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

# --- FastAPI ---
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

# --- Geospatial + Polygon ---
from shapely.geometry import Polygon
import rasterio
import rasterio.features as feat

# --- Optional: TensorFlow (remove if unused) ---
import tensorflow as tf

# --- Custom Modules ---
from cv2geojson import find_geocontours, export_annotations
from extendBoundaryMapBox import extendBoundary  # MapBox Integration

# --- Ensure working directory is set to script location ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from model import Segmenter_segformer

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def overlay_on_image(img, mask_rgb, alpha=0.4):
    """
    Blend only turf and building areas with alpha; keep background pixels unchanged.
    """
    img_np = np.array(img).astype(np.float32)
    m_np = mask_rgb.astype(np.float32)

    # Create boolean mask for blending: True where mask is not background
    blend_mask = ~np.all(mask_rgb == COLORMAP[2], axis=-1)  # shape (H, W)

    # Expand to (H, W, 1) for broadcasting
    blend_mask_exp = np.expand_dims(blend_mask, axis=-1)

    # Perform alpha blending only on relevant pixels
    blended = img_np.copy()
    blended[blend_mask] = (1 - alpha) * img_np[blend_mask] + alpha * m_np[blend_mask]

    return blended.astype(np.uint8)
def opt():
    parser = argparse.ArgumentParser(description='DeepLawn')
    parser.add_argument('--model_path', type=str, default='./models/lawn_model.pth', help='Path to model')
    parser.add_argument('--model_name', type=str, default='mit_b5', help='Segformer model variant')
    parser.add_argument('--mode', type=str, choices=['cpu', 'gpu'], default='gpu', help='CPU or GPU')
    parser.add_argument('--threshold', type=float, default='0.2', help='Threshold for tree model')
    parser.add_argument('--tree_model_path', type=str, default='./models/tree_model.pth', help='Path to tree model')
    
    
    return parser.parse_args()









# Optional helper: convert predicted labels to RGB
def class2color(mask, labels):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label in labels:
        color_mask[mask == label.trainId] = label.color
    return color_mask


def apply_colormap(idx_mask):
    """Convert a 2D index mask to RGB using COLORMAP."""
    h, w = idx_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, col in colormap.items():
        rgb[idx_mask == cls] = col
    return rgb



def tree_test(base_dir,model_path, img_path, image_size, device, model_name='mit_b5'):
    """
    Runs inference using a 2-class tree segmentation model and returns the predicted RGB mask.

    Args:
        model_path (str): Path to the trained 2-class tree model (.pth).
        img_path (str): Path to the input image.
        image_size (int): Input size expected by the model (e.g., 768).
        device (str or torch.device): 'cpu' or 'cuda'.
        model_name (str): Segformer model variant (e.g., 'mit_b5').

    Returns:
        Tuple[None, RGB_mask_image (H, W, 3), ...]
    """
    # Preprocess image
    image = np.array(Image.open(img_path).convert("RGB"))
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    transformed = transform(image=image)
    input_tensor = transformed["image"].unsqueeze(0).to(device).float()  # Shape: [1, 3, H, W]

    # Load model with 2 output classes
    model = Segmenter_segformer(model_name, classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)                        # Shape: (1, 2, H, W)
        probs = torch.softmax(logits, dim=1)               # Convert to probabilities
        preds = torch.argmax(probs, dim=1).squeeze(0)      # Shape: (H, W)

    # Convert predicted mask to RGB format (0=black, 1=green)
    pred_mask = preds.cpu().numpy().astype(np.uint8)
    rgb_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    rgb_mask[pred_mask == 1] = [255, 0, 0]   # Green for trees
    rgb_mask[pred_mask == 0] = [0, 0, 0]     # Black for background


    out_overlay = os.path.join(base_dir, "tree_overlay.png")
    overlay = overlay_on_image(image, rgb_mask, 0.4)
    os.makedirs(os.path.dirname(out_overlay), exist_ok=True)
    Image.fromarray(overlay).save(out_overlay)
    return None, rgb_mask, None, None, None, None

def test(base_dir,model_path, model_name, image_path, IMAGE_SIZE, device, boundary_points=None):

    label = namedtuple('label', ['name', 'color', 'trainId'])
    labels = [label('Building', [255, 100, 0], 0),
              label('turf', [61, 245, 61], 1),
              label('background', [0, 0, 0], 2)]

    start_time = time.time()
    # Load image
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    print(img.size)

    # Resize image for model input
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()  # Converts to tensor and scales to [0, 1]
    ])

    img_org = transform(img_resized)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_resized).unsqueeze(0).to(device)
    model = Segmenter_segformer(model_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        output = torch.nn.functional.interpolate(output, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        preds = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

    # Resize predictions to match original image size
    color_out = class2color(preds, labels) #*255
    #print(color_out.shape)
    color_out = cv2.resize(color_out.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)



    input_mask = np.zeros((color_out.shape[0], color_out.shape[1]), dtype=np.uint8)
    input_points = np.array([boundary_points])
    cv2.fillPoly(input_mask, input_points, 255)
    resultantImage = cv2.bitwise_and(color_out, color_out, mask=input_mask)
    r1 = resultantImage.copy()
    r2 = resultantImage.copy()

    # Ensure folders exist
    os.makedirs('./extracted', exist_ok=True)
    os.makedirs('./images', exist_ok=True)
    os.makedirs('./masks', exist_ok=True)



    # Define unique identifiers for filenames
    timestamp = int(time.time() * 1000)
    unique_id = str(uuid.uuid4())[:8]



    print("Before denorm min/max:", img_org.min(), img_org.max())

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    denorm = img_org * std + mean
    denorm = denorm * 255

    denorm = denorm.cpu().numpy()
    denorm = denorm.clip(0, 255).astype(np.uint8)
    denorm = np.transpose(denorm, (1, 2, 0))  # (C, H, W) → (H, W, C)
    img_path = f'./images/image_{timestamp}_{unique_id}.png'
    Image.fromarray(denorm).save(img_path)


    mask=preds
    color_mask=apply_colormap(mask)
    mask_path = f'./masks/mask_{timestamp}_{unique_id}.png'
    Image.fromarray(color_mask).save(mask_path)


    out_overlay = os.path.join(base_dir, "final_overlay.png")
    #print(np.array(img).shape,color_mask.shape)
    overlay = overlay_on_image(img_resized, color_mask, 0.7)
    os.makedirs(os.path.dirname(out_overlay), exist_ok=True)
    Image.fromarray(overlay).save(out_overlay)


    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Inference took {elapsed:.2f} seconds using device: {device}")


    return img_org, r1, r2,color_out,preds
IMAGE_SIZE = 768
h = w = 768

# Directories setup
os.makedirs("pro", exist_ok=True)
os.makedirs("images", exist_ok=True)

colormap = {
    0: (255, 100, 0),       # building - orange
    1: (0, 255, 0),         # turf - green
    2: (0, 0, 0)            # background - black
}
COLORMAP = {
    0: (61, 245, 61),       # building - green
    1: (255, 100, 0),       # turf - orange
    2: (0, 0, 0)            # background - black
}

# (All helper functions: overlay_on_image, class2color, apply_colormap, tree_test, test, remove_overlapping_polygons remain unchanged)
# ... keep all functions exactly as you provided, no changes needed ...



@app.route('/')
def index():
    return "SERVER IS RUNNNING"



def remove_overlapping_polygons(polygons, overlap_threshold=0.8):
    unique_polygons = []
    polygons_to_remove = set()
    for i, poly1 in enumerate(polygons):
        if i in polygons_to_remove:
            continue
        poly1_shapely = Polygon(poly1['geometry']['coordinates'][0])
        for j, poly2 in enumerate(polygons):
            if i != j and j not in polygons_to_remove:
                poly2_shapely = Polygon(poly2['geometry']['coordinates'][0])
                intersection_area = poly1_shapely.intersection(poly2_shapely).area
                min_area = min(poly1_shapely.area, poly2_shapely.area)
                max_area = max(poly1_shapely.area, poly2_shapely.area)
                # Check if they are duplicates
                if intersection_area / min_area > overlap_threshold and max_area / min_area < 1.1:
                    polygons_to_remove.add(j)
        unique_polygons.append(poly1)
    return unique_polygons

@app.get("/model/predict-boundary")
def predict(
    mapurl: str = Query(...),
    boundaryPoints: str = Query(...),
    features: str = Query(...),
    lat: float = Query(0),
    lon: float = Query(0),
    zoom: float = Query(0)
):
    try:
        print("Received mapUrl:", mapurl)
        print("Received boundaryPointsRaw:", boundaryPoints)
        print("Received featuresRaw:", features)

        try:
            boundaryPoints_json = json.loads(unquote(boundaryPoints))
            features_list = json.loads(unquote(features))
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Invalid input: {e}"})

        try:
            image = imread(mapurl)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Error loading image: {e}"})

        timestamp_milliseconds = int(time.time() * 1000)
        unique_id = str(uuid.uuid4())[:8]
        img_path = f"./images/{timestamp_milliseconds}_{unique_id}_map_image.jpg"

        try:
            imsave(img_path, image)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Error saving image: {e}"})

        session_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        base_dir = f"./predict_boundary/{session_id}"
        os.makedirs(base_dir, exist_ok=True)

        try:
            img, r1, r2, color_out, preds = test(base_dir, model_path, model_name, img_path, IMAGE_SIZE, device, boundaryPoints_json)
            _, tree_mask_image, *_ = tree_test(base_dir, tree_model_path, img_path, IMAGE_SIZE, device)
        except Exception as e:
            os.remove(img_path)
            return JSONResponse(status_code=500, content={"error": f"Processing error: {e}"})

        predictionData = {}

        try:
            if 'LAWN' in features_list:
                mask = np.all(r1 == (61, 245, 61), axis=-1)
                geocontours = find_geocontours(np.array(tf.cast(mask[:, :], tf.int32)), mode='imagej')
                turf_features = [c.export_feature(color=(0, 255, 0), label='roi') for c in geocontours]
                predictionData["LAWN"] = json.dumps(remove_overlapping_polygons(turf_features))

            if 'BUILDING' in features_list:
                mask = np.all(r2 == (255, 100, 0), axis=-1)
                geocontours = find_geocontours(np.array(tf.cast(mask[:, :], tf.int32)), mode='imagej')
                building_features = [c.export_feature(color=(0, 255, 0), label='roi') for c in geocontours]
                predictionData["BUILDING"] = json.dumps(remove_overlapping_polygons(building_features))

            if 'TREE' in features_list:
                tree_mask = np.all(tree_mask_image == (255, 0, 0), axis=-1)
                geocontours = find_geocontours(np.array(tf.cast(tree_mask[:, :], tf.int32)), mode='imagej')
                tree_features = [c.export_feature(color=(255, 0, 0), label='roi') for c in geocontours]
                predictionData["TREE"] = json.dumps(remove_overlapping_polygons(tree_features))

                total_pixels = tree_mask.size
                tree_pixels = np.count_nonzero(tree_mask)
                predictionData["TREE_COVERAGE_PERCENT"] = round((tree_pixels / total_pixels) * 100, 2)

        except Exception as e:
            os.remove(img_path)
            return JSONResponse(status_code=500, content={"error": f"Feature extraction error: {e}"})

        os.remove(img_path)
        return JSONResponse(content=jsonable_encoder(predictionData))

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Unexpected server error: {e}"})

@app.get("/model/extend-boundary/{boundary}")
def extend_boundary_endpoint(boundary: str):
    try:
        temp_boundary = json.loads(boundary)
        extended = extendBoundary(temp_boundary)
        return JSONResponse(content=jsonable_encoder(extended))
    except Exception:
        return JSONResponse(content=jsonable_encoder(temp_boundary))


# Initialize model
args = opt()
model_path = args.model_path
model_name = args.model_name
tree_model_path = args.tree_model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.mode == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    device = torch.device("cpu")



if device.type == 'cuda':
    print("GPU details:", torch.cuda.get_device_properties(0))
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("FASTAPI:app", host="0.0.0.0", port=5000, reload=True)
