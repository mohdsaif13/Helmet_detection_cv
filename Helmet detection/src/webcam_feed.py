# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 19:37:31 2025

@author: hp
"""
import os
import cv2
import time
from datetime import datetime
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO("C:/Users/hp/OneDrive/Desktop/Helmet detection/model/best.pt")

# Try USB camera (0 = default, 1 = external USB, 2 = another USB, etc.)
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)  # change to 0, 1, 2 until your USB cam works

if not cap.isOpened():
    print(" Could not open camera. Check index (0/1/2).")
    exit()
# Get FPS from camera
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps != fps:  # Sometimes returns 0 or NaN
    fps = 10.0
print(" Camera FPS:", fps)

# Create output folder
output_dir = "C:/Users/hp/OneDrive/Desktop/Helmet detection/video_feed"
os.makedirs(output_dir, exist_ok=True)

# Optional: save the video
save_output = True
if save_output:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"webcam_output_{timestamp}.avi")  # mkv format
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID is widely supported
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Run for 15 seconds
start_time = time.time()
print("Recording started")
time.sleep(0.5)
while True:
    success, frame = cap.read()
    if not success:
        print(" Failed to grab frame from camera")
        break

    # Run YOLO detection
    results = model.predict(frame)

    # Annotate results
    annotated_frame = results[0].plot()

    # Show live window
    cv2.imshow("YOLO Helmet Detection", annotated_frame)

    # Save video frame
    if save_output:
        out.write(annotated_frame)

    # Exit on 'q' or after 15 seconds
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("⏹ Exiting on key press")
        break
  
# Release everything
cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
