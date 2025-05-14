import argparse

import cv2
import numpy as np
from rfdetr import RFDETRBase, RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES


def predict_image(model_path, image_path, output_path=None, threshold=0.2, model_type="base"):
    """
    Run RF-DETR prediction on an image
    
    Args:
        model_path: Path to the RF-DETR model weights
        image_path: Path to the input image
        output_path: Path to save the output image with detections (optional)
        threshold: Detection confidence threshold
        model_type: Either "base" (ResNet-50) or "large" (ResNet-101)
    
    Returns:
        List of detection dictionaries with 'box', 'class', and 'score' keys
    """
    # Load model
    if model_type.lower() == "large":
        model = RFDETRLarge(pretrain_weights=model_path)
        print(f"Loaded RF-DETR Large model from {model_path}")
    else:
        model = RFDETRBase(pretrain_weights=model_path)
        print(f"Loaded RF-DETR Base model from {model_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert to RGB (RF-DETR expects RGB input)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run prediction
    from PIL import Image
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"Running RF-DETR prediction on {image_path} with model {model_path}")

    # Convert OpenCV image to PIL image if needed
    if isinstance(image_rgb, np.ndarray):
        pil_image = Image.fromarray(image_rgb)
    else:
        pil_image = image_rgb

    try:
        # Run prediction with error handling
        detections = model.predict(pil_image, threshold=threshold)

        # Log detection details
        if hasattr(detections, 'class_id'):
            detection_count = len(detections.class_id) if hasattr(detections.class_id, '__len__') else 1
            logger.info(f"Detected {detection_count} objects with confidence >= {threshold} (structured format)")
        elif isinstance(detections, list):
            logger.info(f"Detected {len(detections)} objects with confidence >= {threshold} (list format)")
        else:
            logger.info(f"Detection result: {type(detections)}")
    except Exception as e:
        logger.error(f"Error during model prediction: {str(e)}")
        # Return empty detections instead of raising to avoid breaking the UI
        return []

    # Draw results on image
    if output_path:
        image_with_boxes = image.copy()

        # Process detections based on their format
        if hasattr(detections, 'class_id') and hasattr(detections, 'confidence') and hasattr(detections, 'xyxy'):
            # New format with structured attributes
            labels = [
                f"{COCO_CLASSES[class_id]} {confidence:.2f}"
                for class_id, confidence
                in zip(detections.class_id, detections.confidence)
            ]

            for i, (class_id, bbox) in enumerate(zip(detections.class_id, detections.xyxy)):
                class_name = COCO_CLASSES.get(class_id, f"Class {class_id}")
                x1, y1, x2, y2 = map(int, bbox)  # Convert to integers for cv2

                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_with_boxes, labels[i],
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Original dictionary format
            for det in detections:
                box = det['box']
                # Handle different box formats safely
                if hasattr(box, 'tolist') and callable(getattr(box, 'tolist')):
                    # Handle numpy array
                    box_list = box.tolist()
                    x1, y1, x2, y2 = map(int, box_list)
                elif isinstance(box, (list, tuple, np.ndarray)):
                    # Handle list/tuple/array where elements might be numpy arrays
                    try:
                        # Check if individual elements are numpy arrays that need item() extraction
                        if hasattr(box[0], 'item') and callable(getattr(box[0], 'item')):
                            x1 = int(box[0].item())
                            y1 = int(box[1].item())
                            x2 = int(box[2].item())
                            y2 = int(box[3].item())
                        else:
                            # Standard conversion
                            x1 = int(float(box[0]))
                            y1 = int(float(box[1]))
                            x2 = int(float(box[2]))
                            y2 = int(float(box[3]))
                    except (TypeError, ValueError) as e:
                        print(f"Error converting box coordinates: {e}, box: {box}")
                        raise  # Re-raise to handle at higher level
                else:
                    # Unknown format
                    raise ValueError(f"Unexpected box format: {type(box)}")

                # Get class and score
                if 'class_id' in det:
                    class_id = det['class_id']
                    label = COCO_CLASSES.get(class_id, f"Class {class_id}")
                else:
                    label = det.get('class', 'Object')

                score = det.get('score', 1.0)

                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_with_boxes, f"{label}: {score:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imwrite(output_path, image_with_boxes)
        print(f"Saved detection results to {output_path}")

    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RF-DETR object detection on an image")
    parser.add_argument("--model", required=True, help="Path to RF-DETR model weights")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Path to save output image with detections")
    parser.add_argument("--threshold", type=float, default=0.2, help="Detection confidence threshold")
    parser.add_argument("--model-type", choices=["base", "large"], default="base",
                        help="Model type: base (ResNet-50) or large (ResNet-101)")

    args = parser.parse_args()
    predict_image(
        args.model,
        args.image,
        args.output,
        args.threshold,
        args.model_type
    )
