# -*- coding: utf-8 -*-
import os
import numpy as np
import requests

# ‑‑‑ CONFIG ‑‑‑
MAPBOX_TOKEN = "pk.eyJ1Ijoiam9lbG5vcnRocnVwIiwiYSI6ImNtYWI1Njl3ejI5Z24ybHE1ZGFjMG1mM3EifQ.uSax12y2V-qEz-t41_9MtQ"
DIST_FT       = 55                                  # distance threshold in feet
DIST_M        = DIST_FT * 0.3048                    # threshold in metres
FILTER_A_THRESHOLD_M = 1.5                          # distance threshold for point truncation (meters)

def haversine_distance(point1, point2):
    """
    Calculate the great-circle distance between two points on Earth.
    
    Args:
        point1: [lat, lon] coordinates of point 1
        point2: [lat, lon] coordinates of point 2
    
    Returns:
        Distance in meters
    """
    lat1, lon1 = point1
    lat2, lon2 = point2
    
    # Convert to radians
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in meters
    r = 6371000
    
    return c * r

def apply_filter_A(boundary, distance_threshold=FILTER_A_THRESHOLD_M):
    """
    Recursively remove redundant points from the boundary.
    If three consecutive points have distances below threshold between them, 
    the middle point is removed.
    
    Args:
        boundary: List of [lat, lon] points
        distance_threshold: Minimum distance in meters between consecutive points
        
    Returns:
        Filtered boundary with redundant points removed
    """
    if len(boundary) <= 3:
        return boundary
    
    points = np.asarray(boundary, dtype=float)
    keep = np.ones(len(points), dtype=bool)
    changed = False
    
    # Check each triplet of consecutive points
    for i in range(len(points) - 2):
        p1 = points[i]
        p2 = points[i+1]
        p3 = points[i+2]
        
        d1 = haversine_distance(p1, p2)
        d2 = haversine_distance(p2, p3)
        
        if d1 < distance_threshold and d2 < distance_threshold:
            keep[i+1] = False  # Remove middle point
            changed = True
    
    filtered_boundary = points[keep].tolist()
    
    # If points were removed, recursively apply the filter again
    if changed:
        return apply_filter_A(filtered_boundary, distance_threshold)
    else:
        return filtered_boundary

def check_road_distance(point, mapbox_token, dist_m):
    """
    Check if a point is within the threshold distance from a road.
    
    Args:
        point: [lat, lon] coordinates
        mapbox_token: Mapbox API token
        dist_m: Distance threshold in meters
        
    Returns:
        Tuple of (snapped_point, distance, success)
    """
    lat, lon = point
    # Mapbox expects longitude,latitude
    coord_str = f"{lon},{lat};{lon},{lat}"          # duplicate point -> 2‑point "trace"
    radius    = f"{dist_m};{dist_m}"                # radius (m) per coordinate

    url = (f"https://api.mapbox.com/matching/v5/mapbox/driving/{coord_str}.json"
           f"?access_token={mapbox_token}"
           f"&radiuses={radius}"
           f"&steps=false&geometries=geojson")

    try:
        resp = requests.get(url, timeout=2)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[WARN] Request failed: {e}")
        return point, float('inf'), False
    
    if data.get("code") != "Ok" or data["tracepoints"][0] is None:
        return point, float('inf'), False
    
    trace_pt = data["tracepoints"][0]               # first tracepoint ↔ first input coord
    snapped_lon, snapped_lat = trace_pt["location"] # Mapbox returns [lon, lat]
    distance = trace_pt.get("distance", dist_m + 1)  # metres; fallback > threshold
    
    snapped_point = [snapped_lat, snapped_lon]     # return in [lat, lon] format
    
    return snapped_point, distance, True

# ---------------------------------------------------------------------------
#  extendBoundary(boundary)  -->  list[list[lat, lon]]
# ---------------------------------------------------------------------------
def extendBoundary(boundary):
    """
    For every vertex in a property boundary, snap the point to the nearest road
    using Mapbox Map Matching **if** that road is closer than DIST_FT.
    Uses filtering techniques to reduce the number of API calls.
    
    Args:
        boundary: Iterable of [lat, lon] points
        
    Returns:
        List of [lat, lon] points with eligible points snapped to nearby roads
    """
    if MAPBOX_TOKEN is None:
        raise RuntimeError("MAPBOX_TOKEN environment variable not set")

    print(f"Original boundary has {len(boundary)} points")
    
    # Apply Filter A to eliminate redundant points
    reduced_boundary = apply_filter_A(boundary)
    print(f"After Filter A: {len(reduced_boundary)} points (removed {len(boundary) - len(reduced_boundary)})")
    
    # Use numpy array for easier manipulation
    S = np.asarray(reduced_boundary, dtype=float)
    H = len(S)
    S_final = np.zeros_like(S)
    checked = np.zeros(H, dtype=bool)
    road_proximity = np.zeros(H, dtype=bool)  # True if point is near road
    api_calls = 0
    
    # FILTER B - Step 1: Check every other point first
    for i in range(0, H, 2):
        snapped_point, distance, success = check_road_distance(S[i], MAPBOX_TOKEN, DIST_M)
        api_calls += 1
        checked[i] = True
        
        if success and distance <= DIST_M:
            S_final[i] = snapped_point
            road_proximity[i] = True
            print(f"Point {i}: d = {distance:.2f} m → snapped")
        else:
            S_final[i] = S[i]
            road_proximity[i] = False
            print(f"Point {i}: d = {distance:.2f} m → kept")
    
    # FILTER B - Step 2: Look for patterns to skip API calls
    for i in range(0, H - 4, 2):
        # If we have 3 consecutive checked points that are all far from roads
        if (i+4 < H and checked[i] and checked[i+2] and checked[i+4] and
            not road_proximity[i] and not road_proximity[i+2] and not road_proximity[i+4]):
            
            # Assume the points in between are also far from roads
            S_final[i+1] = S[i+1]
            S_final[i+3] = S[i+3]
            checked[i+1] = True
            checked[i+3] = True
            print(f"Point {i+1}: skipped (inferred far from road)")
            print(f"Point {i+3}: skipped (inferred far from road)")
    
    # FILTER B - Step 3: Check remaining points
    for i in range(H):
        if not checked[i]:
            snapped_point, distance, success = check_road_distance(S[i], MAPBOX_TOKEN, DIST_M)
            api_calls += 1
            
            if success and distance <= DIST_M:
                S_final[i] = snapped_point
                print(f"Point {i}: d = {distance:.2f} m → snapped")
            else:
                S_final[i] = S[i]
                print(f"Point {i}: d = {distance:.2f} m → kept")
    
    print(f"Total Mapbox API calls: {api_calls} (original would use {len(boundary)})")
    return S_final.tolist()
