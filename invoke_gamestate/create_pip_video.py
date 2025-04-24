import os
import cv2
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from visualization.visualizer import get_unique_frame_ids, get_frame_info, FrameInfo

def create_pip_video(original_video_path, minimap_video_path, output_path, pip_scale=0.3, target_resolution=(1280, 720)):
    """
    Create a picture-in-picture video with the original video as the main content
    and the minimap video in the lower right corner.
    
    Args:
        original_video_path: Path to the original video
        minimap_video_path: Path to the minimap video
        output_path: Path to save the output video
        pip_scale: Scale factor for the picture-in-picture (minimap) video (default: 0.3)
        target_resolution: Resolution to scale both videos to before creating PiP (default: 1280x720)
    """
    # Open both videos
    cap_original = cv2.VideoCapture(original_video_path)
    cap_minimap = cv2.VideoCapture(minimap_video_path)
    
    # Check if videos opened successfully
    if not cap_original.isOpened() or not cap_minimap.isOpened():
        print(f"Error: Could not open one or both videos.")
        return
    
    # Get properties of both videos
    width_original = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_original = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width_minimap = int(cap_minimap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_minimap = int(cap_minimap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_original.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use target resolution for both videos
    target_width, target_height = target_resolution
    
    # Calculate the size for the minimap PiP (after both are scaled to same resolution)
    width_pip = int(target_width * pip_scale)
    height_pip = int(target_height * pip_scale)
    
    # Calculate the position for the minimap PiP (lower right corner)
    pos_x = target_width - width_pip - 20  # 20 pixels from the right edge
    pos_y = target_height - height_pip - 20  # 20 pixels from the bottom edge
    
    # Create a VideoWriter for the output video
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec - highly compatible
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    # Process frame by frame
    with tqdm(total=frame_count, desc=f"Creating PiP video for {os.path.basename(original_video_path)}") as pbar:
        while True:
            ret_original, frame_original = cap_original.read()
            ret_minimap, frame_minimap = cap_minimap.read()
            
            # If either video ends, break the loop
            if not ret_original or not ret_minimap:
                break
            
            # Scale both videos to the same target resolution
            frame_original = cv2.resize(frame_original, (target_width, target_height))
            frame_minimap = cv2.resize(frame_minimap, (target_width, target_height))
            
            # Resize the minimap frame for PiP
            minimap_resized = cv2.resize(frame_minimap, (width_pip, height_pip))
            
            # Add minimap to the original frame (in the lower right corner)
            # Create an ROI in the original frame
            roi = frame_original[pos_y:pos_y+height_pip, pos_x:pos_x+width_pip]
            
            # Add a background for the PiP
            cv2.rectangle(frame_original, (pos_x-2, pos_y-2), (pos_x+width_pip+2, pos_y+height_pip+2), (0, 0, 0), -1)
            
            # Put the minimap in the ROI
            frame_original[pos_y:pos_y+height_pip, pos_x:pos_x+width_pip] = minimap_resized
            
            # Write the frame to the output video
            out.write(frame_original)
            
            pbar.update(1)
    
    # Release resources
    cap_original.release()
    cap_minimap.release()
    out.release()
    
    print(f"PiP video created: {output_path}")

def get_referee_detected_videos():
    """Get list of videos where a referee is detected."""
    json_files = os.listdir("data/gamestate_output")
    
    referee_detected_videos = []
    for json_file in json_files:
        json_file_path = f"data/gamestate_output/{json_file}"
        if "action" not in json_file or not os.path.exists(json_file_path):
            continue
            
        filename = Path(json_file_path).stem
        frame_ids = get_unique_frame_ids(json_file_path)
        
        if not frame_ids:
            continue
            
        # Check if referee is detected
        referee_detected = False
        for frame_id in frame_ids:
            frame_info = get_frame_info(json_file_path, frame_id)
            for player in frame_info.players:
                if player.role == "referee":
                    referee_detected = True
                    break
            if referee_detected:
                break
                
        if referee_detected:
            referee_detected_videos.append(filename)
            
    return referee_detected_videos

def process_all_videos(output_dir="visualization/output_pip", pip_scale=0.3, target_resolution=(1280, 720)):
    """Process all videos with referee detection and create PiP videos."""
    # Get list of videos with referee detection
    referee_detected_videos = get_referee_detected_videos()
    print(f"Found {len(referee_detected_videos)} videos with referee detection")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each video
    for action_name in tqdm(referee_detected_videos, desc="Processing videos"):
        original_video_path = f"data/Dataset/Train/{action_name}/clip_0.mp4"
        minimap_video_path = f"visualization/output/{action_name}/video/{action_name}.mp4"
        output_path = f"{output_dir}/{action_name}_pip.mp4"
        
        # Check if both videos exist
        if not os.path.exists(original_video_path) or not os.path.exists(minimap_video_path):
            print(f"Skipping {action_name}: one or both videos not found")
            continue
            
        create_pip_video(original_video_path, minimap_video_path, output_path, pip_scale, target_resolution)

if __name__ == "__main__":
    process_all_videos() 