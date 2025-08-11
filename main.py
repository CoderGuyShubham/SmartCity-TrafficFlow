import cv2
import numpy as np

# Define HSV ranges for red, yellow, and green
red_lower = np.array([170, 100, 100])
red_upper = np.array([180, 255, 255])
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([30, 255, 255])
green_lower = np.array([70, 100, 100])
green_upper = np.array([80, 255, 255])


def track_vehicles(frame, previous_centroids):
  # Convert to HSV for background subtraction
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  background_model = cv2.BackgroundSubtractorMOG2()  # Create background model
  foreground = background_model.apply(hsv)

  # Detect moving objects using foreground mask
  thresh = cv2.threshold(foreground, 25, 255, cv2.THRESH_BINARY)[1]
  kernel = np.ones((5, 5), np.uint8)
  opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
  closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

  # Find contours and update centroids
  current_centroids = []
  for contour in cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
    if cv2.contourArea(contour) > 500:
      M = cv2.moments(contour)
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])
      current_centroids.append((cX, cY))

      # Match with previous centroids (replace with Kalman filter for better tracking)
      if previous_centroids is not None:
        for i, (prevX, prevY) in enumerate(previous_centroids):
          dist = ((cX - prevX) ** 2 + (cY - prevY) ** 2) ** 0.5
          if dist < 50:  # Adjust threshold for matching distance
            previous_centroids[i] = (cX, cY)
            break

  return frame, current_centroids

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

vehicle_classes = ["car", "bus", "truck"]

def detect_vehicles(frame):
    height, width, _ = frame.shape

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Run forward pass
    outputs = net.forward(output_layers)

    vehicles = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # COCO class IDs for car=2, motorcycle=3, bus=5, truck=7
            if class_id in [2, 3, 5, 7] and confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Apply Non-Max Suppression to remove duplicates
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            vehicles.append((x, y, w, h))
            # Draw green bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Vehicle", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, vehicles

def detect_traffic_lights(frame):
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # Mask for each color
  red_mask = cv2.inRange(hsv, red_lower, red_upper)
  yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
  green_mask = cv2.inRange(hsv, green_lower, green_upper)

  # Find contours for each color
  red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Draw contours and identify dominant color
  for contour in red_contours:
    if cv2.contourArea(contour) > 100:  # Adjust threshold for size
      cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)
      light_state = "RED"
      break
  else:
    for contour in yellow_contours:
      if cv2.contourArea(contour) > 100:
        cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)
        light_state = "YELLOW"
        break
    else:
      for contour in green_contours:
        
        if cv2.contourArea(contour) > 100:
          cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
          light_state = "GREEN"
          break
      else:
        light_state = "UNKNOWN"

  return frame, light_state
cap = cv2.VideoCapture('video.mp4')

if not cap.isOpened():
    print("Error: Cannot open video source")
    exit()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to grab frame. Check video path or camera connection.")
        break

    frame, vehicles = detect_vehicles(frame)
    resized_img = cv2.resize(frame, None, fx=0.2, fy=0.2)
    cv2.imshow("Traffic Light Detection", resized_img)
    print(f"Detected vehicles: {len(vehicles)}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()