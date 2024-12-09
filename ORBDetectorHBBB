import cv2
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

# Initialize ORB for fallback
orb = cv2.ORB_create()

# Load PyTorch models
dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
dino_model.eval()

detr_model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
detr_model.eval()

# Preprocessing for DINO
preprocess_dino = transforms.Compose([
    transforms.Resize((224, 224)),  # Explicit size to avoid deprecation warnings
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_dino_features(image):
    """
    Extract features from an image using DINO.
    """
    image = Image.fromarray(image)
    if image.size[0] == 0 or image.size[1] == 0:
        raise ValueError(f"Image has invalid dimensions: {image.size}")
    image_tensor = preprocess_dino(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = dino_model(image_tensor)
    return features

def detect_objects_with_detr(image):
    """
    Detect objects in the input image using DETR.
    """
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = detr_model(image_tensor)

    probabilities = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    boxes = outputs['pred_boxes'][0]
    return probabilities, boxes

def match_keypoints(des_template, des_frame):
    """
    Match descriptors using BFMatcher with Lowe's ratio test.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des_template, des_frame, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return good_matches

def find_homography_and_draw(template, kp_template, frame, kp_frame, good_matches):
    """
    Find homography and draw bounding box if enough matches are found.
    """
    if len(good_matches) > 20:  # Minimum matches threshold
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = template.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
        print("Match found using homography!")
    return frame

# Load template image and extract DINO features
template_path = '../../images/won_1000.jpg'
template_image = cv2.imread(template_path)
if template_image is None:
    raise FileNotFoundError(f"Template image not found at {template_path}")
template_features = extract_dino_features(template_image)

# Extract ORB keypoints and descriptors for fallback
template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
kp_template, des_template = orb.detectAndCompute(template_gray, None)

# Access the camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert frame to RGB for PyTorch models
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Step 1: Detect objects with DETR
        probabilities, boxes = detect_objects_with_detr(frame_rgb)
        confidence_threshold = 0.7
        for prob, box in zip(probabilities, boxes):
            if prob.max().item() > confidence_threshold:
                # Extract object region
                box = box.cpu().numpy()
                x_min, y_min, x_max, y_max = box * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                if y_max <= y_min or x_max <= x_min:
                    print(f"Invalid bounding box: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
                    continue
                object_region = frame_rgb[y_min:y_max, x_min:x_max]

                # Step 2: Extract DINO features for the detected region
                if object_region.size == 0:
                    print("Skipping empty object region")
                    continue

                object_features = extract_dino_features(object_region)

                # Step 3: Match DINO features with template features
                similarity_score = torch.nn.functional.cosine_similarity(template_features, object_features).item()
                if similarity_score > 0.8:
                    print("Detected and matched object using DINO!")
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Step 4: Fallback to ORB if DINO/DETR fails
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = orb.detectAndCompute(frame_gray, None)
        if des_frame is not None:
            good_matches = match_keypoints(des_template, des_frame)
            frame = find_homography_and_draw(template_gray, kp_template, frame, kp_frame, good_matches)

        # Display the frame
        cv2.imshow('Integrated Pipeline', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
