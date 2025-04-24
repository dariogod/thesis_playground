import cv2
import os

# Create frames directory if it doesn't exist
frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)

# Open the video file
video = cv2.VideoCapture("input.mp4")

frame_count = 0

while True:
    # Read next frame
    success, frame = video.read()
    
    if not success:
        break
        
    # Save frame as JPG file
    frame_path = os.path.join(frames_dir, f"frame_{(frame_count+1):04d}.jpg")
    cv2.imwrite(frame_path, frame)
    
    frame_count += 1

# Release video capture object
video.release()

print(f"Extracted {frame_count} frames to {frames_dir}/")
