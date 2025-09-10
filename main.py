import cv2
import numpy as np
import time

# ------------------- Load YOLO -------------------
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Detect these classes
vehicle_classes = ["car", "bus", "truck", "motorbike", "bicycle"]


def detect_vehicles(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    vehicles, confidences, boxes, class_ids = [], [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in vehicle_classes:
                center_x, center_y = int(detection[0]*width), int(detection[1]*height)
                w, h = int(detection[2]*width), int(detection[3]*height)
                x, y = int(center_x - w/2), int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            vehicles.append((x,y,w,h,class_ids[i]))
            label = f"{classes[class_ids[i]]} {int(confidences[i]*100)}%"
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, label, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    return frame, vehicles


# ------------------- Load 4 Cameras -------------------
cams = {
    "North": cv2.VideoCapture("video.mp4"),
    "East": cv2.VideoCapture("video2.mp4"),
    "South": cv2.VideoCapture("video3.webm"),
    "West": cv2.VideoCapture("video4.mp4")
}
for name in cams.keys():
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)


# ------------------- Smart Controller -------------------
cycle = ["North", "East", "South", "West"]
idx = 0  # start with North
green_start_time = time.time()


def compute_green_time(count):
    # Base time = 5 sec, 1 sec per 2 vehicles, max 30 sec
    base = 5
    extra = count // 2
    return min(base + extra, 30)


while True:
    traffic_counts = {}

    # Step 1: detect vehicles in all cameras
    for direction, cap in cams.items():
        ret, frame = cap.read()
        if not ret:
            continue

        frame, vehicles = detect_vehicles(frame)
        traffic_counts[direction] = len(vehicles)

        # Show vehicle count only
        cv2.putText(frame, f"{direction} Vehicles: {len(vehicles)}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.imshow(direction, frame)

    if not traffic_counts:
        break

    # Step 2: non-blocking timer for green signal
    current_dir = cycle[idx]
    count = traffic_counts.get(current_dir, 0)
    green_time = compute_green_time(count)
    elapsed = time.time() - green_start_time

    if elapsed >= green_time:
        # Move to next direction in cycle
        idx = (idx + 1) % len(cycle)
        current_dir = cycle[idx]
        green_start_time = time.time()

    # Step 3: print signals in console
    print("-------------------------------------------------")
    for d in cycle:
        if d == current_dir:
            remaining = max(0, int(green_time - elapsed))
            print(f"🟢 {d}: GREEN for {remaining}s (Vehicles={traffic_counts.get(d,0)})")
        else:
            print(f"🔴 {d}: RED   (Vehicles={traffic_counts.get(d,0)})")
    print("-------------------------------------------------")

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ------------------- Cleanup -------------------
for cap in cams.values():
    cap.release()
cv2.destroyAllWindows()
