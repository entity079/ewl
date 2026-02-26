import json
import os
import numpy as np
from PIL import Image
import cv2  # for polygon to mask
from tqdm import tqdm
import shutil

def coco_to_instance_masks(coco_json_path, image_dir, output_label_dir, images_output_dir, file_ending='.png'):
    """
    Convert COCO instance segmentation annotations to nnU-Net label images and prepare images for imagesTr.

    :param coco_json_path: Path to COCO JSON annotation file.
    :param image_dir: Directory containing the images.
    :param output_label_dir: Directory to save the label images.
    :param images_output_dir: Directory to save the renamed images for imagesTr.
    :param file_ending: File extension for labels, e.g., '.png'
    """
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create a mapping from image_id to image info
    image_id_to_info = {img['id']: img for img in coco_data['images']}

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    sorted_image_ids = sorted(annotations_by_image.keys(), key=lambda x: int(x))
    image_id_to_case_index = {img_id: i for i, img_id in enumerate(sorted_image_ids)}

    for img_id, anns in tqdm(annotations_by_image.items(), desc="Processing images"):
        img_info = image_id_to_info[img_id]
        img_filename = img_info['file_name']
        img_path = os.path.join(image_dir, img_filename)

        case_index = image_id_to_case_index[img_id]
        case_id = f"case_{case_index:04d}"

        # Load image to get dimensions
        image = Image.open(img_path)
        width, height = image.size  # assuming PIL, for PNG

        # Initialize label image
        label_img = np.zeros((height, width), dtype=np.uint16)  # uint16 for many instances

        instance_id = 1  # start from 1, background 0
        for ann in anns:
            if 'segmentation' in ann:
                seg = ann['segmentation']
                if isinstance(seg, list) and len(seg) > 0:
                    # Polygon
                    if isinstance(seg[0], list):
                        # Multiple polygons
                        mask = np.zeros((height, width), dtype=np.uint8)
                        for poly in seg:
                            poly = np.array(poly).reshape(-1, 2).astype(int)
                            cv2.fillPoly(mask, [poly], 1)
                        label_img[mask == 1] = instance_id
                    else:
                        # Single polygon
                        poly = np.array(seg).reshape(-1, 2).astype(int)
                        cv2.fillPoly(label_img, [poly], instance_id)
                elif isinstance(seg, dict):
                    # RLE mask
                    # Need pycocotools for decoding RLE
                    from pycocotools import mask as mask_utils
                    rle = mask_utils.frPyObjects(seg, img_info['height'], img_info['width'])
                    mask = mask_utils.decode(rle)
                    label_img[mask > 0] = instance_id
                else:
                    print(f"Unsupported segmentation format for annotation {ann['id']}")
                    continue
            instance_id += 1

        # Save label image
        label_filename = case_id + file_ending
        label_path = os.path.join(output_label_dir, label_filename)
        Image.fromarray(label_img.astype(np.uint16)).save(label_path)

        # Copy and rename image to images_output_dir
        _, ext = os.path.splitext(img_filename)
        new_img_name = case_id + '_0000' + ext
        shutil.copy(img_path, os.path.join(images_output_dir, new_img_name))

if __name__ == "__main__":
    # Example usage
    coco_json_path = r"D:\\intern\\cag\\arcade\\syntax\\train\\annotations\\train.json"  # Update this
    image_dir = r"D:\\intern\\cag\\arcade\\syntax\\train\\images"  # Update this if different
    output_label_dir = r"D:\\intern\\cag\\arcade\\syntax\\train\\labelsTr"  # Update this
    images_output_dir = r"D:\\intern\\cag\\arcade\\syntax\\train\\imagesTr"  # Update this
    coco_to_instance_masks(coco_json_path, image_dir, output_label_dir, images_output_dir)
