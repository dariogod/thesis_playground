import cv2

# Open input video
cap = cv2.VideoCapture('input_soccer_clip.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create output video writer with h264 codec
out = cv2.VideoWriter('input_soccer_clip_h264.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Write frame with new codec
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
