import json
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any

def convert_coco_to_masks(
    coco_json_path: str, 
    image_dir: str
):
    print(f"Loading COCO annotations from {coco_json_path}...")
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
        
    # Build dictionaries for easy lookup
    images_by_id = {img['id']: img for img in coco_data['images']}
    
    # Initialize a dict to hold polygons for each image
    polygons_by_image = {img['id']: [] for img in coco_data['images']}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        segmentation = ann['segmentation']
        
        for poly in segmentation:
            # Poly is a list of [x1, y1, x2, y2, ...]
            # Convert to numpy array of shape (N, 1, 2)
            poly_arr = np.array(poly).reshape(-1, 2).astype(np.int32)
            polygons_by_image[img_id].append(poly_arr)
            
    print("Generating masks and renaming images...")
    processed = 0
    missing = 0
    
    # Search for all available original image files in the image_dir
    available_images = list(Path(image_dir).rglob("*.jpg"))
    available_image_names = {path.name: path for path in available_images}
    
    for img_id, img_info in images_by_id.items():
        original_filename = img_info['file_name']
        
        # Label studio prefixes a hash, like "...\\1234abcd-DJI_0529_frame_0120.jpg"
        # We just want the 'DJI_...' part
        base_name = os.path.basename(original_filename)
        if "-" in base_name:
            # Strip the label studio UUID prefix
            base_name = base_name.split("-", 1)[1]
            
        # Find the original image path
        if base_name not in available_image_names:
            print(f"Warning: Could not find {base_name} in {image_dir}")
            missing += 1
            continue
            
        img_path = available_image_names[base_name]
        
        # We must rename it to match the dataset.py format which expects '*_sat.jpg'
        # E.g., DJI_0529_frame_0120.jpg -> DJI_0529_frame_0120_sat.jpg
        new_base_name = base_name.replace(".jpg", "_sat.jpg")
        new_img_path = img_path.with_name(new_base_name)
        
        # Only rename if it hasn't been renamed yet
        if img_path != new_img_path and img_path.exists():
            img_path.rename(new_img_path)
            
        # Now create the mask image: '*_mask.png'
        mask_base_name = base_name.replace(".jpg", "_mask.png")
        mask_path = img_path.with_name(mask_base_name)
        
        width = img_info['width']
        height = img_info['height']
        
        # Create black mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Fill polygons with white (255)
        polygons = polygons_by_image[img_id]
        if polygons:
            cv2.fillPoly(mask, polygons, color=255)
            
        cv2.imwrite(str(mask_path), mask)
        processed += 1
        
    print(f"\nSuccessfully generated {processed} masks and renamed images.")
    if missing > 0:
        print(f"Failed to find {missing} original images.")

if __name__ == "__main__":
    coco_json_path = r"C:\Users\svish\Downloads\Testing model 1\deepglobe-road-segmentation\project-2-at-2026-09-30-17-15-2ccec4b5\result.json"
    image_dir = r"C:\Users\svish\Downloads\Testing model 1\deepglobe-road-segmentation\data\manual_annotation\images"
    convert_coco_to_masks(coco_json_path, image_dir)
