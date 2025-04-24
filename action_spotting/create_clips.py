import json
import os
import cv2
import numpy as np
from pathlib import Path

# Get the script directory to build absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Event dictionary
EVENT_DICTIONARY_V2 = {
    "Penalty": 0, 
    "Kick-off": 1, 
    "Goal": 2, 
    "Substitution": 3, 
    "Offside": 4, 
    "Shots on target": 5,
    "Shots off target": 6, 
    "Clearance": 7, 
    "Ball out of play": 8, 
    "Throw-in": 9, 
    "Foul": 10,
    "Indirect free-kick": 11, 
    "Direct free-kick": 12, 
    "Corner": 13, 
    "Yellow card": 14, 
    "Red card": 15, 
    "Yellow->red card": 16
}

# Path to the match directory - use absolute paths
MATCH_DIR = os.path.join(SCRIPT_DIR, "data/england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal")
FRAMES_DIR = os.path.join(MATCH_DIR, "frames")
RESULTS_FILE = os.path.join(MATCH_DIR, "results_spotting.json")
CLIPS_DIR = os.path.join(MATCH_DIR, "clips")

print(f"Match directory: {MATCH_DIR}")
print(f"Frames directory: {FRAMES_DIR}")
print(f"Results file: {RESULTS_FILE}")
print(f"Clips directory: {CLIPS_DIR}")

# Create clips directory if it doesn't exist
os.makedirs(CLIPS_DIR, exist_ok=True)

# Check if the results file exists
if not os.path.exists(RESULTS_FILE):
    print(f"Error: Results file not found at {RESULTS_FILE}")
    exit(1)

# Load detection results
try:
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)
except Exception as e:
    print(f"Error loading results file: {e}")
    exit(1)

# Create matrix for storing frame confidences
amount_of_classes = len(EVENT_DICTIONARY_V2)
amount_of_frames = 0
for item in data:
    if item["frame"] > amount_of_frames:
        amount_of_frames = item["frame"]
amount_of_frames += 1

matrix = np.zeros((amount_of_frames, amount_of_classes))

for item in data:
    frame = item["frame"]
    label = item["label"]
    label_id = EVENT_DICTIONARY_V2[label]
    confidence = item["score"]
    
    matrix[frame, label_id] = confidence

# Function to create a clip for a detected event
def create_clip(center_frame, event_name, confidence, clip_length=126*2):
    # Calculate start and end frames
    half_length = clip_length // 2
    start_frame = max(0, center_frame - half_length)
    end_frame = min(amount_of_frames - 1, center_frame + half_length)
    
    # Adjust start frame if needed to ensure we have clip_length frames
    if end_frame - start_frame + 1 < clip_length and start_frame > 0:
        start_frame = max(0, end_frame - clip_length + 1)
    
    # Adjusted end frame if needed
    if end_frame - start_frame + 1 < clip_length and end_frame < amount_of_frames - 1:
        end_frame = min(amount_of_frames - 1, start_frame + clip_length - 1)
    
    # Sanitize event name for filename
    sanitized_event_name = event_name.replace(" ", "_").replace("->", "to")
    
    # Create clip filename
    clip_filename = f"{center_frame}_{sanitized_event_name}_{confidence:.3f}.mp4"
    clip_path = os.path.join(CLIPS_DIR, clip_filename)
    
    print(f"Creating clip: {clip_path}")
    print(f"  Using frames from {start_frame} to {end_frame}")
    
    # Get first frame to determine dimensions
    # Ensure we're using an even frame number
    first_frame_num = start_frame if start_frame % 2 == 0 else start_frame + 1
    first_frame_path = os.path.join(FRAMES_DIR, f"frame{first_frame_num}.jpg")
    
    if not os.path.exists(first_frame_path):
        print(f"Error: Frame file not found at {first_frame_path}")
        return
    
    first_frame = cv2.imread(first_frame_path)
    
    if first_frame is None:
        print(f"Error: Could not read frame {first_frame_path}")
        return
    
    height, width, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(clip_path, fourcc, 12.5, (width, height))
    
    # Add frames to video - only use even frames as they're the only ones available
    frames_added = 0
    for frame_idx in range(start_frame, end_frame + 1):
        # Skip odd frame numbers as they don't exist
        if frame_idx % 2 != 0:
            continue
            
        frame_path = os.path.join(FRAMES_DIR, f"frame{frame_idx}.jpg")
        
        if not os.path.exists(frame_path):
            print(f"Warning: Frame file not found at {frame_path}")
            continue
            
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
        
        out.write(frame)
        frames_added += 1
    
    # Release video writer
    out.release()
    print(f"Clip created: {clip_path} with {frames_added} frames")

# Process all detected events with confidence >= 0.8
print(f"Looking for events with confidence >= 0.8...")
clips_created = 0

for frame in range(amount_of_frames):
    for label_id in range(amount_of_classes):
        confidence = matrix[frame, label_id]
        if confidence >= 0.8:
            # Get event name by looking up label_id in EVENT_DICTIONARY_V2
            event_name = [k for k,v in EVENT_DICTIONARY_V2.items() if v == label_id][0]
            create_clip(frame, event_name, confidence)
            clips_created += 1

print(f"Finished creating {clips_created} clips.") 