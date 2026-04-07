import os
import cv2

def extract_frames(video_path, output_dir, fps=5):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    interval = int(frame_rate / fps)

    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            cv2.imwrite(f"{output_dir}/frame_{saved}.jpg", frame)
            saved += 1

        count += 1

    cap.release()
