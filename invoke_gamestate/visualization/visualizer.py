#!/usr/bin/env python
import argparse
import json
import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import math
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm
from .pitch_drawer import (
    FrameInfo, Player, Position, 
    draw_frame, euclidean, distance_to_line, 
    get_point_of_action, get_closest_point_on_line
)

def analyze_ref_position(frame_info: FrameInfo) -> FrameInfo:
    """
    Analyze the referee's position relative to players and ideal positioning.
    Updates the frame_info object with the analysis results.
    
    Args:
        frame_info: Frame information including players and referee
        
    Returns:
        FrameInfo: The updated frame information with analysis data
    """
    # Extract referee and player positions
    ref_pos = None
    player_positions = []
    
    for player in frame_info.players:
        if player.role == "referee":
            ref_pos = player.position
        elif player.role in ["player", "goalkeeper"]:
            player_positions.append(player.position)
    
    # If no referee or players found, return the original frame_info
    if not ref_pos or not player_positions:
        return frame_info
    
    # Calculate point of action (player centroid)
    point_of_action = get_point_of_action(player_positions)
    
    # Calculate distances
    dist_to_diagonal = distance_to_line(ref_pos)
    dist_to_action = euclidean(ref_pos, point_of_action)
    
    # Update frame_info with analysis data
    frame_info.point_of_action = point_of_action
    frame_info.dist_to_diagonal = dist_to_diagonal
    frame_info.dist_to_point_of_action = dist_to_action
    
    return frame_info

def get_frame_info(json_file_path: str, frame_id: Optional[str] = None) -> FrameInfo:
    """
    Extract information about a specific frame from a JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing predictions
        frame_id: Specific frame ID to extract. If None, uses the first available frame.
        
    Returns:
        FrameInfo: The extracted frame information
    """
    # Load JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Get unique image IDs
    image_ids = set()
    if "predictions" in data:
        for pred in data["predictions"]:
            if "image_id" in pred:
                image_ids.add(pred["image_id"])
    
    # Determine frame ID to extract
    selected_image_id = None
    if frame_id and frame_id in image_ids:
        selected_image_id = frame_id
    else:
        if frame_id and frame_id not in image_ids:
            print(f"Warning: Frame ID '{frame_id}' not found. Using first available frame.")
        selected_image_id = list(sorted(image_ids))[0] if image_ids else ""
    
    # Extract players from the SNGS format
    players = []
    if "predictions" in data:
        for pred in data["predictions"]:
            if pred.get("image_id") == selected_image_id:
                # Skip if no bbox_pitch
                if "bbox_pitch" not in pred:
                    continue
                
                # Get attributes
                attributes = pred.get("attributes", {})
                role = attributes.get("role", "player")
                team = attributes.get("team", "unknown")
                jersey = attributes.get("jersey", None)
                
                # Create player
                player = Player(
                    role=role,
                    team=team,
                    jersey_number=jersey,
                    position=Position(
                        x=pred["bbox_pitch"]["x_bottom_middle"],
                        y=pred["bbox_pitch"]["y_bottom_middle"]
                    )
                )
                players.append(player)
    
    frame_info = FrameInfo(frame_id=selected_image_id, players=players)
    
    # Analyze referee position
    frame_info = analyze_ref_position(frame_info)
    
    return frame_info

def visualize(json_file_path: str, output_path: Optional[str] = None, show: bool = False, 
              pitch_scale: float = 3, line_thickness: int = 3, frame_id: Optional[str] = None) -> Tuple[np.ndarray, FrameInfo]:
    """
    Visualize predictions from a JSON file on a soccer pitch.
    
    Args:
        json_file_path: Path to JSON file containing predictions
        output_path: Path to save the visualization. If None, the image is not saved
        show: Whether to display the visualization
        pitch_scale: Scale factor for the pitch
        line_thickness: Thickness of lines
        frame_id: Specific frame ID to visualize
        
    Returns:
        Tuple[np.ndarray, FrameInfo]: The visualization image and frame information with analysis
    """
    # Get frame information with analysis
    frame_info = get_frame_info(json_file_path, frame_id)
    
    # Draw the frame using the function from pitch_drawer
    image = draw_frame(frame_info, pitch_scale, line_thickness)
    
    # Save the image if output_path is provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)
    
    # Show the image if requested
    if show:
        cv2.namedWindow("Pitch Visualization", cv2.WINDOW_NORMAL)
        cv2.imshow("Pitch Visualization", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return image, frame_info

def create_video(frame_paths: List[Path], output_path: Path, fps: int = 10) -> None:
    """
    Create a video from a list of image frames.
    
    Args:
        frame_paths: List of paths to frame images
        output_path: Path to save the output video
        fps: Frames per second for the output video
    """
    if not frame_paths:
        print("Error: No frames provided")
        return
    
    print(f"Creating video from {len(frame_paths)} frames...")
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(str(frame_paths[0]))
    height, width, _ = first_frame.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Add frames to video with tqdm progress bar
    for frame_path in tqdm(frame_paths, desc="Creating video"):
        frame = cv2.imread(str(frame_path))
        video_writer.write(frame)
    
    # Release video writer
    video_writer.release()
    print(f"Video created successfully: {output_path}")

def get_unique_frame_ids(json_file_path: str) -> List[str]:
    """
    Get all unique frame IDs from a JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing predictions
        
    Returns:
        List[str]: Sorted list of unique frame IDs
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    image_ids = set()
    if "predictions" in data:
        for pred in data["predictions"]:
            if "image_id" in pred:
                image_ids.add(pred["image_id"])
    
    return sorted(image_ids)
