import cv2
import numpy as np
import time
import pandas as pd
from datetime import datetime

# ------------------- Load YOLO -------------------
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
outs = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in outs]

vehicle_classes = ["car", "bus", "truck", "motorbike", "bicycle"]

# ------------------- Vehicle Detection -------------------
def detect_vehicles(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    vehicles, confidences, boxes, class_ids = [], [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in vehicle_classes:
                center_x, center_y = int(detection[0]*width), int(detection[1]*height)
                w, h = int(detection[2]*width), int(detection[3]*height)
                x, y = int(center_x - w/2), int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    idxs = indexes.flatten() if len(indexes) > 0 else []

    for i in range(len(boxes)):
        if i in idxs:
            x, y, w, h = boxes[i]
            vehicles.append((x, y, w, h, class_ids[i]))
            label = f"{classes[class_ids[i]]} {int(confidences[i]*100)}%"
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, label, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame, vehicles

# ------------------- Vehicle Logging -------------------
history = []

def log_detection(direction, vehicles):
    for v in vehicles:
        history.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "direction": direction,
            "vehicle": classes[v[4]]
        })

# ------------------- Overlay Helper -------------------
def put_text_with_bg(img, text, org, font, font_scale, thickness,
                     text_color, bg_color=(0,0,0), alpha=0.55, pad=6):
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    rect_x1 = max(0, x - pad)
    rect_y1 = max(0, y - text_h - pad)
    rect_x2 = min(img.shape[1], x + text_w + pad)
    rect_y2 = min(img.shape[0], y + baseline + pad)
    overlay = img.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    cv2.putText(img, text, (x,y), font, font_scale, text_color, thickness, cv2.LINE_AA)

# ------------------- Load Cameras -------------------
cams = {
    "North": cv2.VideoCapture("video.mp4"),
    "East": cv2.VideoCapture("video2.mp4"),
    "South": cv2.VideoCapture("video3.webm"),
    "West": cv2.VideoCapture("video4.mp4")
}
for name in cams.keys():
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)

last_frames = {d: None for d in cams.keys()}
last_logged_time = {d: 0 for d in cams.keys()}
log_interval = 1.0  # seconds
last_csv_save = time.monotonic()
csv_save_interval = 5.0  # seconds

# ------------------- Traffic Controller -------------------
cycle = ["North", "East", "South", "West"]
idx = 0
current_dir = cycle[idx]
traffic_counts = {d:0 for d in cycle}
green_time = 10
green_remaining = green_time
last_tick = time.monotonic()

def compute_green_time(count):
    return min(5 + count//2, 30)

def compute_wait_until_target(cycle, current_idx, target_idx, traffic_counts, remaining_current):
    if target_idx == current_idx:
        return 0
    wait = remaining_current
    i = (current_idx+1) % len(cycle)
    while i != target_idx:
        wait += compute_green_time(traffic_counts.get(cycle[i],0))
        i = (i+1) % len(cycle)
    return wait

# ------------------- Main Loop -------------------
while True:
    frames = {}
    any_frame = False
    now_loop = time.monotonic()

    for d, cap in cams.items():
        if d == current_dir:
            ret, frame = cap.read()
            if not ret:
                frames[d] = last_frames[d]
                continue
            any_frame = True
            frame, vehicles = detect_vehicles(frame)
            traffic_counts[d] = len(vehicles)

            if now_loop - last_logged_time[d] >= log_interval:
                log_detection(d, vehicles)
                last_logged_time[d] = now_loop

            last_frames[d] = frame
            frames[d] = frame
        else:
            if last_frames[d] is not None:
                frames[d] = last_frames[d]
            else:
                ret, frame = cap.read()
                if ret:
                    last_frames[d] = frame
                    frames[d] = frame
                else:
                    frames[d] = None

    if not any_frame:
        break

    # countdown
    now = time.monotonic()
    if now - last_tick >= 1.0:
        green_remaining -= 1
        last_tick = now

    # switch lane
    if green_remaining <= 0:
        idx = (idx + 1) % len(cycle)
        current_dir = cycle[idx]
        green_time = compute_green_time(traffic_counts.get(current_dir,0))
        green_remaining = green_time
        last_tick = now

    # overlay
    for d in cycle:
        frame = frames.get(d)
        if frame is None:
            continue
        scale_factor = frame.shape[1]/640.0
        font_scale = max(0.5, 0.9*scale_factor)
        thickness = max(1,int(round(2*scale_factor)))
        font = cv2.FONT_HERSHEY_SIMPLEX
        x = int(20*scale_factor)
        y_vehicle = int(40*scale_factor)
        y_signal = int(80*scale_factor)

        put_text_with_bg(frame,
            f"{d} Vehicles: {traffic_counts.get(d,0)}",
            (x,y_vehicle), font, font_scale, thickness,
            text_color=(255,255,255),
            bg_color=(0,0,0), alpha=0.6)

        if d == current_dir:
            put_text_with_bg(frame,
                f"GREEN {green_remaining}s",
                (x,y_signal), font, font_scale, thickness,
                text_color=(255,255,255),
                bg_color=(0,150,0), alpha=0.7)
        else:
            target_idx = cycle.index(d)
            wait_seconds = compute_wait_until_target(cycle, idx, target_idx, traffic_counts, green_remaining)
            put_text_with_bg(frame,
                f"RED Wait {wait_seconds}s",
                (x,y_signal), font, font_scale, thickness,
                text_color=(255,255,255),
                bg_color=(90,0,0), alpha=0.7)

        cv2.imshow(d, frame)

    # save CSV every 5 seconds
    if now_loop - last_csv_save >= csv_save_interval and len(history) > 0:
        pd.DataFrame(history).to_csv("traffic_log.csv", index=False)
        last_csv_save = now_loop

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# final save
if len(history) > 0:
    pd.DataFrame(history).to_csv("traffic_log.csv", index=False)
    print("✅ traffic_log.csv saved successfully")

# cleanup
for cap in cams.values():
    cap.release()
cv2.destroyAllWindows()
