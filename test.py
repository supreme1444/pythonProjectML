import os

import cv2

video_path = '1_1WM8HGDW.mov'
output_folder = 'frames/'
os.makedirs(output_folder, exist_ok=True)


cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    exit()
frame_count = 0
saved_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 15 == 0:
        cv2.imwrite(f"{output_folder} frame {frame_count}.jpg", frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Ок {frame_count} кадров, сохранено {saved_count}.")
