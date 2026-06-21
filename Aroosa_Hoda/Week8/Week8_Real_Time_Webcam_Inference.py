import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy']

NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CONFIDENCE_THRESHOLD = 0.60

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_mobilenet_v2(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    feature_dim = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(feature_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


def load_model(weights_path: str) -> nn.Module:
    print(f"Building MobileNetV2 architecture ({NUM_CLASSES} classes)...")
    model = build_mobilenet_v2(NUM_CLASSES)

    print(f"Loading weights from {weights_path} ...")
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}.")
    return model

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def preprocess_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    tensor = preprocess(resized)
    tensor = tensor.unsqueeze(0)
    return tensor.to(DEVICE)


@torch.no_grad()
def predict(model: nn.Module, frame_bgr: np.ndarray):
    x = preprocess_frame(frame_bgr)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
    return label, float(probs[idx]), probs


def draw_prediction(frame, label, confidence, fps, uncertain: bool):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), thickness=-1)

    if uncertain:
        text = f"Uncertain ({confidence*100:.1f}%)"
        color = (0, 165, 255)
    else:
        text = f"{label} ({confidence*100:.1f}%)"
        color = (0, 255, 0)

    cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}  |  Device: {DEVICE.type}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    bar_w = int((w - 20) * confidence)
    cv2.rectangle(frame, (10, h - 25), (10 + bar_w, h - 10), color, thickness=-1)
    cv2.rectangle(frame, (10, h - 25), (w - 10, h - 10), (255, 255, 255), thickness=1)

    return frame


def main(weights_path: str, camera_index: int, infer_every_n: int):
    model = load_model(weights_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Try a different index (0, 1, 2...) or check OS camera permissions."
        )

    print("Webcam started. Press 'q' to quit, 's' to save a snapshot.")

    frame_times = deque(maxlen=30)
    frame_count = 0
    last_label, last_conf = "warming up...", 0.0
    uncertain = True

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame — exiting.")
                break

            if frame_count % infer_every_n == 0:
                last_label, last_conf, _ = predict(model, frame)
                uncertain = last_conf < CONFIDENCE_THRESHOLD

            frame_times.append(time.time() - t0)
            fps = 1.0 / (sum(frame_times) / len(frame_times))

            display_frame = draw_prediction(frame.copy(), last_label, last_conf, fps, uncertain)
            cv2.imshow("Plant Disease Classifier (PyTorch) — Week 8", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(fname, frame)
                print(f"Saved {fname}")

            frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time plant disease classifier (Week 8, PyTorch)")
    parser.add_argument("--weights", type=str, required=True,
                         help="Path to the .pth state_dict saved in Week 7")
    parser.add_argument("--camera", type=int, default=0,
                         help="Camera index (default 0). Try 1 or 2 if 0 doesn't work.")
    parser.add_argument("--infer-every-n", type=int, default=5,
                         help="Run inference every N frames (default 5).")
    args = parser.parse_args()

    main(args.weights, args.camera, args.infer_every_n)
