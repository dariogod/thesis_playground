import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
import math
from pydantic import BaseModel
from sklearn.cluster import KMeans

class Position(BaseModel):
    x: float
    y: float

class Player(BaseModel):
    role: str = "player"
    team: str = "unknown"
    jersey_number: Optional[int] = None
    position: Position

class FrameInfo(BaseModel):
    frame_id: str
    players: List[Player] = []
    # Referee analysis fields
    point_of_action: Optional[Position] = None
    dist_to_diagonal: Optional[float] = None
    dist_to_point_of_action: Optional[float] = None

def create_pitch_image(image_width: int = 1920, image_height: int = 1080, bg_color: Tuple[int, int, int] = (0, 100, 0)) -> np.ndarray:
    """
    Create a blank image with background color for the pitch.
    
    Args:
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        bg_color: Background color in BGR format
        
    Returns:
        np.ndarray: The initialized image
    """
    # Create a blank image
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    
    # Fill the entire background with green
    image[:] = bg_color
    
    return image

def calculate_pitch_dimensions(image_width: int, image_height: int, pitch_width: float = 105, pitch_height: float = 68, margin_factor: float = 0.9) -> dict:
    """
    Calculate pitch dimensions to fit the screen.
    
    Args:
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        pitch_width: Width of the pitch in meters
        pitch_height: Height of the pitch in meters
        margin_factor: Factor to leave margin around the pitch (0.9 = 90% of screen)
        
    Returns:
        dict: Dictionary containing pitch dimensions and positions
    """
    # Calculate scale to fit pitch on screen while maintaining aspect ratio
    width_scale = image_width / pitch_width
    height_scale = image_height / pitch_height
    pitch_scale = min(width_scale, height_scale) * margin_factor
    
    # Calculate centered position
    radar_center_x = int(image_width/2)
    radar_center_y = int(image_height/2)
    radar_width = int(pitch_width * pitch_scale)
    radar_height = int(pitch_height * pitch_scale)
    radar_top_x = int(radar_center_x - radar_width / 2)
    radar_top_y = int(radar_center_y - radar_height / 2)
    
    return {
        "pitch_scale": pitch_scale,
        "radar_center_x": radar_center_x,
        "radar_center_y": radar_center_y,
        "radar_width": radar_width,
        "radar_height": radar_height,
        "radar_top_x": radar_top_x,
        "radar_top_y": radar_top_y
    }

def draw_pitch(image: np.ndarray, dimensions: dict, line_thickness: int = 2, line_color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    """
    Draw a soccer pitch on the given image.
    
    Args:
        image: Image to draw on
        dimensions: Dictionary containing pitch dimensions from calculate_pitch_dimensions
        line_thickness: Thickness of the lines
        line_color: Color of the lines in BGR format
        
    Returns:
        None (modifies the image in-place)
    """
    # Extract dimensions
    pitch_scale = dimensions["pitch_scale"]
    radar_center_x = dimensions["radar_center_x"]
    radar_center_y = dimensions["radar_center_y"]
    radar_width = dimensions["radar_width"]
    radar_height = dimensions["radar_height"]
    radar_top_x = dimensions["radar_top_x"]
    radar_top_y = dimensions["radar_top_y"]
    
    # Draw field lines
    # Center line
    cv2.line(image, 
            (radar_center_x, radar_top_y), 
            (radar_center_x, radar_top_y + radar_height), 
            line_color, line_thickness)
    
    # Draw pitch outline
    cv2.rectangle(image, 
                (radar_top_x, radar_top_y), 
                (radar_top_x + radar_width, radar_top_y + radar_height), 
                line_color, line_thickness)
    
    # Center circle
    cv2.circle(image, 
            (int(radar_center_x), int(radar_center_y)), 
            int(9.15 * pitch_scale), 
            line_color, line_thickness)
    
    # Center spot
    cv2.circle(image, 
            (int(radar_center_x), int(radar_center_y)), 
            int(0.5 * pitch_scale), 
            line_color, -1)
    
    # Goal areas - 5.5m from each goal post and 5.5m into the field (total width of 18.32m)
    goal_area_width = 5.5 * pitch_scale
    goal_area_height = 18.32 * pitch_scale  # 9.16m on each side of the center
    
    # Left goal area
    left_goal_x = radar_top_x
    cv2.rectangle(image, 
                (left_goal_x, int(radar_center_y - goal_area_height/2)), 
                (int(left_goal_x + goal_area_width), int(radar_center_y + goal_area_height/2)), 
                line_color, line_thickness)
    
    # Right goal area
    right_goal_x = radar_top_x + radar_width - int(goal_area_width)
    cv2.rectangle(image, 
                (right_goal_x, int(radar_center_y - goal_area_height/2)), 
                (radar_top_x + radar_width, int(radar_center_y + goal_area_height/2)), 
                line_color, line_thickness)
    
    # Penalty areas - 16.5m from each goal post and 16.5m into the field (total width of 40.32m)
    penalty_area_width = 16.5 * pitch_scale
    penalty_area_height = 40.32 * pitch_scale  # 20.16m on each side of the center
    
    # Left penalty area
    cv2.rectangle(image, 
                (radar_top_x, int(radar_center_y - penalty_area_height/2)), 
                (int(radar_top_x + penalty_area_width), int(radar_center_y + penalty_area_height/2)), 
                line_color, line_thickness)
    
    # Right penalty area
    cv2.rectangle(image, 
                (radar_top_x + radar_width - int(penalty_area_width), int(radar_center_y - penalty_area_height/2)),
                (radar_top_x + radar_width, int(radar_center_y + penalty_area_height/2)), 
                line_color, line_thickness)
    
    # Penalty spots - 11m from goal line
    penalty_spot_distance = 11 * pitch_scale
    
    # Left penalty spot
    left_penalty_spot_x = int(radar_top_x + penalty_spot_distance)
    cv2.circle(image, 
            (left_penalty_spot_x, int(radar_center_y)), 
            int(0.5 * pitch_scale), 
            line_color, -1)
    
    # Right penalty spot
    right_penalty_spot_x = int(radar_top_x + radar_width - penalty_spot_distance)
    cv2.circle(image, 
            (right_penalty_spot_x, int(radar_center_y)), 
            int(0.5 * pitch_scale), 
            line_color, -1)
    
    # Penalty arcs - 9.15m radius from penalty spot
    penalty_arc_radius = 9.15 * pitch_scale
    
    # Left penalty arc
    cv2.ellipse(image, 
                (left_penalty_spot_x, int(radar_center_y)), 
                (int(penalty_arc_radius), int(penalty_arc_radius)),
                0, 127 + 180, 233 + 180,  # Draw only the part outside the penalty area
                line_color, line_thickness)
    
    # Right penalty arc
    cv2.ellipse(image, 
                (right_penalty_spot_x, int(radar_center_y)), 
                (int(penalty_arc_radius), int(penalty_arc_radius)),
                0, 127, 233,  # Draw only the part outside the penalty area
                line_color, line_thickness)
    
    # Corner arcs - 1m radius
    corner_radius = 1 * pitch_scale
    
    # Top-left corner
    cv2.ellipse(image, 
                (int(radar_top_x), int(radar_top_y)), 
                (int(corner_radius), int(corner_radius)),
                0, 0, 90, 
                line_color, line_thickness)
    
    # Top-right corner
    cv2.ellipse(image, 
                (int(radar_top_x + radar_width), int(radar_top_y)), 
                (int(corner_radius), int(corner_radius)),
                0, 90, 180, 
                line_color, line_thickness)
    
    # Bottom-left corner
    cv2.ellipse(image, 
                (int(radar_top_x), int(radar_top_y + radar_height)), 
                (int(corner_radius), int(corner_radius)),
                0, 270, 360, 
                line_color, line_thickness)
    
    # Bottom-right corner
    cv2.ellipse(image, 
                (int(radar_top_x + radar_width), int(radar_top_y + radar_height)), 
                (int(corner_radius), int(corner_radius)),
                0, 180, 270, 
                line_color, line_thickness)

def get_pitch_coordinates(dimensions: dict, x: float, y: float) -> Tuple[int, int]:
    """
    Convert pitch coordinates to image coordinates.
    
    Args:
        dimensions: Dictionary containing pitch dimensions
        x: X coordinate on the pitch (center of field is 0,0)
        y: Y coordinate on the pitch (center of field is 0,0)
        
    Returns:
        Tuple[int, int]: (x, y) coordinates on the image
    """
    radar_x = dimensions["radar_center_x"] + int(x * dimensions["pitch_scale"])
    radar_y = dimensions["radar_center_y"] + int(y * dimensions["pitch_scale"])
    
    return radar_x, radar_y

# Utility functions for referee position analysis
def euclidean(p1: Position, p2: Position) -> float:
    """Calculate Euclidean distance between two positions."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def distance_to_line(point: Position, dimensions: dict) -> float:
    """Calculate distance from a point to the diagonal line."""
    # Get the pitch corners
    corners = get_pitch_corners(dimensions)
    
    # Get diagonal endpoints (bottom-left to top-right)
    p1_x, p1_y = corners["bottom_left"]
    p2_x, p2_y = corners["top_right"]
    
    # Calculate the line equation parameters: y = mx + b
    # Slope (m) = (y2 - y1) / (x2 - x1)
    m = (p2_y - p1_y) / (p2_x - p1_x)
    
    # y-intercept (b) = y1 - m * x1
    b = p1_y - m * p1_x
    
    # Distance formula
    numerator = abs(m * point.x - point.y + b)
    denominator = math.sqrt(m**2 + 1)
    return numerator / denominator

def get_point_of_action(player_positions: List[Position], k=1) -> Position:
    """Calculate the centroid of player positions using KMeans clustering."""
    if not player_positions:
        return Position(x=0.0, y=0.0)
    X = np.array([[p.x, p.y] for p in player_positions])
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    return Position(x=float(kmeans.cluster_centers_[0][0]), 
                    y=float(kmeans.cluster_centers_[0][1]))

def get_closest_point_on_diagonal(point: Position, dimensions: dict) -> Position:
    """Get the closest point on the diagonal line to a given point."""
    # Get the pitch corners
    corners = get_pitch_corners(dimensions)
    
    # Get diagonal endpoints (bottom-left to top-right)
    p1_x, p1_y = corners["bottom_left"]
    p2_x, p2_y = corners["top_right"]
    
    # Calculate the line equation parameters: y = mx + b
    # Slope (m) = (y2 - y1) / (x2 - x1)
    m = (p2_y - p1_y) / (p2_x - p1_x)
    
    # y-intercept (b) = y1 - m * x1
    b = p1_y - m * p1_x
    
    # For line y = mx + b, the closest point to (x0, y0) is:
    x0, y0 = point.x, point.y
    x = (x0 + m * y0 - m * b) / (1 + m * m)
    y = m * x + b
    
    return Position(x=x, y=y)

def get_pitch_corners(dimensions: dict) -> Dict[str, Tuple[float, float]]:
    """
    Get the pitch corners in pitch coordinates.
    
    Args:
        dimensions: Dictionary containing pitch dimensions
        
    Returns:
        Dict: Dictionary with corner coordinates in pitch coordinate system
    """
    pitch_width = dimensions["radar_width"] / dimensions["pitch_scale"]
    pitch_height = dimensions["radar_height"] / dimensions["pitch_scale"]
    half_width = pitch_width / 2
    half_height = pitch_height / 2
    
    corners = {
        "top_left": (-half_width, -half_height),
        "top_right": (half_width, -half_height),
        "bottom_left": (-half_width, half_height),
        "bottom_right": (half_width, half_height)
    }
    
    return corners

def draw_diagonal(image: np.ndarray, dimensions: dict, line_thickness: int = 2, line_color: Tuple[int, int, int] = (200, 200, 200)) -> None:
    """
    Draw the diagonal line from bottom-left to top-right corner of the pitch.
    
    Args:
        image: Image to draw on
        dimensions: Dictionary containing pitch dimensions
        line_thickness: Thickness of the line
        line_color: Color of the line in BGR format
        
    Returns:
        None (modifies the image in-place)
    """
    corners = get_pitch_corners(dimensions)
    
    # Get bottom left and top right corners
    bottom_left = corners["bottom_left"]
    top_right = corners["top_right"]
    
    # Convert to image coordinates
    x1, y1 = get_pitch_coordinates(dimensions, bottom_left[0], bottom_left[1])
    x2, y2 = get_pitch_coordinates(dimensions, top_right[0], top_right[1])
    
    # Draw the diagonal line
    cv2.line(image, (x1, y1), (x2, y2), line_color, line_thickness)

def draw_player_markers(image: np.ndarray, frame_info: FrameInfo, dimensions: dict) -> None:
    """
    Draw player markers on the pitch visualization.
    
    Args:
        image: Image to draw on
        frame_info: Frame information including players
        dimensions: Dictionary containing pitch dimensions
        
    Returns:
        None (modifies the image in-place)
    """
    # Team colors
    left_color = (0, 0, 255)  # Blue
    right_color = (255, 0, 0)  # Red
    unknown_color = (255, 255, 255)  # White
    
    # Plot players
    for player in frame_info.players:
        # Get player position
        x = player.position.x
        y = player.position.y
        
        # Convert pitch coordinates to image coordinates
        radar_x, radar_y = get_pitch_coordinates(dimensions, x, y)
        
        # Determine color based on team
        if player.role == "goalkeeper":
            color = (0, 0, 0)  # Black for goalkeepers
        elif player.role == "referee":
            color = (0, 255, 255)  # Yellow for referees
        elif player.team == "left":
            color = left_color
        elif player.team == "right":
            color = right_color
        elif player.role == "ball":
            color = (128, 128, 128)  # Grey for ball
        else:
            color = unknown_color
        
        # Draw player marker (just a circle)
        cv2.circle(
            image,
            (radar_x, radar_y),
            int(dimensions["pitch_scale"] / 2),  # Smaller circle size
            color=color,
            thickness=-1
        )

def draw_analysis_elements(image: np.ndarray, frame_info: FrameInfo, dimensions: dict) -> None:
    """
    Draw referee analysis elements on the pitch visualization.
    
    Args:
        image: Image to draw on
        frame_info: Frame information including players and analysis data
        dimensions: Dictionary containing pitch dimensions
        
    Returns:
        None (modifies the image in-place)
    """
    # Find referee position
    ref_pos = None
    for player in frame_info.players:
        if player.role == "referee":
            ref_pos = player.position
            break
            
    # Draw point of action (player centroid) if available
    if frame_info.point_of_action:
        point_of_action = frame_info.point_of_action
        
        # Convert point of action to image coordinates
        poa_x, poa_y = get_pitch_coordinates(dimensions, point_of_action.x, point_of_action.y)
        
        # Draw the point of action as a cross
        cross_size = int(dimensions["pitch_scale"])
        cv2.line(image, (poa_x - cross_size, poa_y), (poa_x + cross_size, poa_y), (0, 255, 0), 2)
        cv2.line(image, (poa_x, poa_y - cross_size), (poa_x, poa_y + cross_size), (0, 255, 0), 2)
    
    # If we have a referee and analysis data, draw the distances
    if ref_pos and frame_info.point_of_action and frame_info.dist_to_diagonal is not None:
        # Convert ref position to image coordinates
        ref_x, ref_y = get_pitch_coordinates(dimensions, ref_pos.x, ref_pos.y)
        
        # Get closest point on diagonal line to referee
        closest_diag_point = get_closest_point_on_diagonal(ref_pos, dimensions)
        diag_x, diag_y = get_pitch_coordinates(dimensions, closest_diag_point.x, closest_diag_point.y)
        
        # Get point of action coordinates
        poa_x, poa_y = get_pitch_coordinates(dimensions, frame_info.point_of_action.x, frame_info.point_of_action.y)
        
        # Draw line from ref to diagonal
        cv2.line(image, (ref_x, ref_y), (diag_x, diag_y), (200, 200, 200), 2)
        
        # Draw line from ref to point of action
        cv2.line(image, (ref_x, ref_y), (poa_x, poa_y), (0, 255, 0), 2)
        
        # Add distance labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        
        # Midpoint for diagonal distance
        mid_diag_x = (ref_x + diag_x) // 2
        mid_diag_y = (ref_y + diag_y) // 2
        cv2.putText(image, f"{frame_info.dist_to_diagonal:.1f}m", (mid_diag_x, mid_diag_y), 
                   font, font_scale, (255, 255, 255), 2)
        
        # Midpoint for action point distance
        mid_poa_x = (ref_x + poa_x) // 2
        mid_poa_y = (ref_y + poa_y) // 2
        cv2.putText(image, f"{frame_info.dist_to_point_of_action:.1f}m", (mid_poa_x, mid_poa_y), 
                   font, font_scale, (255, 255, 255), 2)

def draw_frame(frame_info: FrameInfo, pitch_scale: float = 3, line_thickness: int = 3) -> np.ndarray:
    """
    Draw a visualization of a frame on a soccer pitch.
    
    Args:
        frame_info: Frame information including players and analysis data
        pitch_scale: Scale factor for the pitch
        line_thickness: Thickness of lines
        
    Returns:
        np.ndarray: The visualization image
    """
    # Create a blank image with green background
    image_width, image_height = 1920, 1080
    image = create_pitch_image(image_width, image_height)
    
    # Calculate pitch dimensions
    dimensions = calculate_pitch_dimensions(image_width, image_height)
    
    # Draw the pitch on the image
    draw_pitch(image, dimensions, line_thickness)
    
    # Draw the diagonal line (from bottom-left to top-right)
    draw_diagonal(image, dimensions, line_thickness)
    
    # Draw player markers
    draw_player_markers(image, frame_info, dimensions)
    
    # Draw analysis elements
    draw_analysis_elements(image, frame_info, dimensions)
    
    return image 