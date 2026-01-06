import argparse
from enum import Enum
from typing import Iterator, List, Optional
import time

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration
from stream_processor import RTMPStreamProcessor

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def load_model(model_path: str, device: str) -> YOLO:
    """
    Load a YOLO model, supporting both .pt and .engine (TensorRT) formats.
    
    Args:
        model_path: Path to model file (.pt or .engine)
        device: Device to load model on ('cpu', 'cuda', etc.)
        
    Returns:
        Loaded YOLO model
    """
    # Check if TensorRT engine exists for this model
    if model_path.endswith('.pt'):
        engine_path = model_path.replace('.pt', '-fp16.engine')
        if os.path.exists(engine_path):
            print(f"Using TensorRT engine: {engine_path}")
            return YOLO(engine_path, task='detect')
    
    # Load regular model
    return YOLO(model_path).to(device=device)


def resize_frame_if_needed(frame: np.ndarray, max_resolution: Optional[int]) -> np.ndarray:
    """
    Resize frame if it exceeds maximum resolution while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        max_resolution: Maximum dimension (width or height)
        
    Returns:
        Resized frame or original frame if no resize needed
    """
    if max_resolution is None:
        return frame
    
    h, w = frame.shape[:2]
    max_dim = max(h, w)
    
    if max_dim > max_resolution:
        scale = max_resolution / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h))
    
    return frame


class RealtimeTeamClassifier:
    """
    Real-time team classifier using running statistics instead of two-pass processing.
    """
    def __init__(self, device: str = 'cpu'):
        self.team_classifier = TeamClassifier(device=device)
        self.is_fitted = False
        self.crop_buffer = []
        self.warmup_frames = 30  # Number of frames to collect before fitting
        
    def update_and_predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Update classifier with new crops and predict team IDs.
        
        Args:
            crops: List of player image crops
            
        Returns:
            Array of team IDs
        """
        if not crops:
            return np.array([])
        
        # Collect crops for initial fitting
        if not self.is_fitted:
            self.crop_buffer.extend(crops)
            
            if len(self.crop_buffer) >= self.warmup_frames:
                print(f"Fitting team classifier with {len(self.crop_buffer)} crops...")
                self.team_classifier.fit(self.crop_buffer)
                self.is_fitted = True
                self.crop_buffer = []  # Clear buffer
                print("Team classifier fitted successfully")
        
        # Use default classification if not fitted yet
        if not self.is_fitted:
            # Return alternating team IDs as placeholder
            return np.array([i % 2 for i in range(len(crops))])
        
        # Predict team IDs
        return self.team_classifier.predict(crops)


def create_stream_frame_generator(
    stream_processor: RTMPStreamProcessor,
    max_resolution: Optional[int] = None
) -> Iterator[np.ndarray]:
    """
    Create a frame generator that reads from RTMP stream.
    
    Args:
        stream_processor: RTMPStreamProcessor instance
        max_resolution: Optional maximum resolution for frames
        
    Yields:
        Frames from the stream
    """
    while True:
        frame = stream_processor.read_frame(timeout=1.0)
        if frame is None:
            continue
        
        if max_resolution:
            frame = resize_frame_if_needed(frame, max_resolution)
        
        yield frame


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
    return radar


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        yield annotated_frame


def run_team_classification(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame


def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels,
            custom_color_lookup=color_lookup)

        h, w, _ = frame.shape
        radar = render_radar(detections, keypoints, color_lookup)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        yield annotated_frame


def run_team_classification_realtime(
    frame_generator: Iterator[np.ndarray],
    device: str
) -> Iterator[np.ndarray]:
    """
    Run team classification in real-time mode (single-pass) for streaming.
    
    Args:
        frame_generator: Generator yielding video frames
        device: Device to run the model on
        
    Yields:
        Annotated frames with team classification
    """
    player_detection_model = load_model(PLAYER_DETECTION_MODEL_PATH, device)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    realtime_classifier = RealtimeTeamClassifier(device=device)
    
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = realtime_classifier.update_and_predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        if len(players) > 0 and len(goalkeepers) > 0:
            goalkeepers_team_id = resolve_goalkeepers_team_id(
                players, players_team_id, goalkeepers)
        else:
            goalkeepers_team_id = np.array([])

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame


def run_radar_realtime(
    frame_generator: Iterator[np.ndarray],
    device: str
) -> Iterator[np.ndarray]:
    """
    Run radar mode in real-time (single-pass) for streaming.
    
    Args:
        frame_generator: Generator yielding video frames
        device: Device to run the model on
        
    Yields:
        Annotated frames with radar view
    """
    player_detection_model = load_model(PLAYER_DETECTION_MODEL_PATH, device)
    pitch_detection_model = load_model(PITCH_DETECTION_MODEL_PATH, device)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    realtime_classifier = RealtimeTeamClassifier(device=device)
    
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = realtime_classifier.update_and_predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        if len(players) > 0 and len(goalkeepers) > 0:
            goalkeepers_team_id = resolve_goalkeepers_team_id(
                players, players_team_id, goalkeepers)
        else:
            goalkeepers_team_id = np.array([])

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels,
            custom_color_lookup=color_lookup)

        h, w, _ = frame.shape
        radar = render_radar(detections, keypoints, color_lookup)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        yield annotated_frame


def run_streaming_mode(
    input_rtmp_url: str,
    output_rtmp_url: str,
    device: str,
    mode: Mode,
    realtime: bool,
    max_resolution: Optional[int]
) -> None:
    """
    Run video processing in streaming mode with RTMP input/output.
    
    Args:
        input_rtmp_url: RTMP URL for input stream
        output_rtmp_url: RTMP URL for output stream
        device: Device to run models on
        mode: Processing mode
        realtime: Use real-time single-pass processing
        max_resolution: Maximum resolution for input frames
    """
    print(f"Starting stream processor...")
    print(f"  Input: {input_rtmp_url}")
    print(f"  Output: {output_rtmp_url}")
    print(f"  Mode: {mode.value}")
    print(f"  Device: {device}")
    print(f"  Real-time: {realtime}")
    print(f"  Max resolution: {max_resolution}")
    
    # Create stream processor
    stream_processor = RTMPStreamProcessor(
        input_rtmp_url=input_rtmp_url,
        output_rtmp_url=output_rtmp_url,
        max_queue_size=30,
        reconnect_delay=5,
        fps=30
    )
    
    try:
        # Start streaming threads
        stream_processor.start()
        
        # Create frame generator from stream
        stream_frame_gen = create_stream_frame_generator(stream_processor, max_resolution)
        
        # Create processing frame generator based on mode
        if mode == Mode.PITCH_DETECTION:
            pitch_detection_model = load_model(PITCH_DETECTION_MODEL_PATH, device)
            for frame in stream_frame_gen:
                result = pitch_detection_model(frame, verbose=False)[0]
                keypoints = sv.KeyPoints.from_ultralytics(result)
                annotated_frame = frame.copy()
                annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
                    annotated_frame, keypoints, CONFIG.labels)
                stream_processor.write_frame(annotated_frame)
                
        elif mode == Mode.PLAYER_DETECTION:
            player_detection_model = load_model(PLAYER_DETECTION_MODEL_PATH, device)
            for frame in stream_frame_gen:
                result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(result)
                annotated_frame = frame.copy()
                annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
                annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
                stream_processor.write_frame(annotated_frame)
                
        elif mode == Mode.BALL_DETECTION:
            ball_detection_model = load_model(BALL_DETECTION_MODEL_PATH, device)
            ball_tracker = BallTracker(buffer_size=20)
            ball_annotator = BallAnnotator(radius=6, buffer_size=10)
            
            def callback(image_slice: np.ndarray) -> sv.Detections:
                result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
                return sv.Detections.from_ultralytics(result)
            
            slicer = sv.InferenceSlicer(
                callback=callback,
                overlap_filter_strategy=sv.OverlapFilter.NONE,
                slice_wh=(640, 640),
            )
            
            for frame in stream_frame_gen:
                detections = slicer(frame).with_nms(threshold=0.1)
                detections = ball_tracker.update(detections)
                annotated_frame = frame.copy()
                annotated_frame = ball_annotator.annotate(annotated_frame, detections)
                stream_processor.write_frame(annotated_frame)
                
        elif mode == Mode.PLAYER_TRACKING:
            player_detection_model = load_model(PLAYER_DETECTION_MODEL_PATH, device)
            tracker = sv.ByteTrack(minimum_consecutive_frames=3)
            for frame in stream_frame_gen:
                result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(result)
                detections = tracker.update_with_detections(detections)
                labels = [str(tracker_id) for tracker_id in detections.tracker_id]
                annotated_frame = frame.copy()
                annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
                annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
                    annotated_frame, detections, labels=labels)
                stream_processor.write_frame(annotated_frame)
                
        elif mode == Mode.TEAM_CLASSIFICATION:
            if realtime:
                frame_generator = run_team_classification_realtime(stream_frame_gen, device)
            else:
                raise ValueError("TEAM_CLASSIFICATION requires --realtime flag in streaming mode")
            
            for annotated_frame in frame_generator:
                stream_processor.write_frame(annotated_frame)
                
        elif mode == Mode.RADAR:
            if realtime:
                frame_generator = run_radar_realtime(stream_frame_gen, device)
            else:
                raise ValueError("RADAR requires --realtime flag in streaming mode")
            
            for annotated_frame in frame_generator:
                stream_processor.write_frame(annotated_frame)
        else:
            raise NotImplementedError(f"Mode {mode} is not implemented for streaming.")
        
        # Print performance stats periodically
        last_stats_time = time.time()
        while True:
            current_time = time.time()
            if current_time - last_stats_time >= 5:  # Every 5 seconds
                stats = stream_processor.get_performance_stats()
                print(f"Performance - Input: {stats['input_fps']:.1f} FPS, "
                      f"Output: {stats['output_fps']:.1f} FPS, "
                      f"Latency: {stats['latency_ms']:.1f}ms")
                last_stats_time = current_time
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping stream processor...")
    finally:
        stream_processor.stop()
        print("Stream processor stopped")


def main(
    source_video_path: Optional[str],
    target_video_path: Optional[str],
    device: str,
    mode: Mode,
    stream_mode: bool = False,
    input_rtmp_url: Optional[str] = None,
    output_rtmp_url: Optional[str] = None,
    realtime: bool = False,
    max_resolution: Optional[int] = None
) -> None:
    """
    Main function to run video processing in file or streaming mode.
    
    Args:
        source_video_path: Path to source video file
        target_video_path: Path to output video file
        device: Device to run models on
        mode: Processing mode
        stream_mode: Enable RTMP streaming mode
        input_rtmp_url: RTMP URL for input stream (streaming mode)
        output_rtmp_url: RTMP URL for output stream (streaming mode)
        realtime: Use real-time single-pass processing
        max_resolution: Maximum resolution for input frames
    """
    # Streaming mode
    if stream_mode:
        if not input_rtmp_url or not output_rtmp_url:
            raise ValueError("Streaming mode requires --input_rtmp_url and --output_rtmp_url")
        
        run_streaming_mode(
            input_rtmp_url=input_rtmp_url,
            output_rtmp_url=output_rtmp_url,
            device=device,
            mode=mode,
            realtime=realtime,
            max_resolution=max_resolution
        )
        return
    
    # File mode (original behavior)
    if not source_video_path or not target_video_path:
        raise ValueError("File mode requires --source_video_path and --target_video_path")
    
    # Apply max_resolution if specified
    if max_resolution:
        print(f"Note: --max_resolution is only applied in streaming mode for file-based processing")
    
    if mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.TEAM_CLASSIFICATION:
        if realtime:
            # Use real-time mode for file processing too
            file_frame_gen = sv.get_video_frames_generator(source_path=source_video_path)
            frame_generator = run_team_classification_realtime(file_frame_gen, device)
        else:
            frame_generator = run_team_classification(
                source_video_path=source_video_path, device=device)
    elif mode == Mode.RADAR:
        if realtime:
            # Use real-time mode for file processing too
            file_frame_gen = sv.get_video_frames_generator(source_path=source_video_path)
            frame_generator = run_radar_realtime(file_frame_gen, device)
        else:
            frame_generator = run_radar(
                source_video_path=source_video_path, device=device)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Soccer AI - Video Analysis and RTMP Streaming',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # File-based processing (original behavior)
  python main.py --source_video_path data/2e57b9_0.mp4 \\
                 --target_video_path output.mp4 \\
                 --device cuda --mode PLAYER_TRACKING
  
  # RTMP streaming mode
  python main.py --stream_mode \\
                 --input_rtmp_url rtmp://localhost/live/input \\
                 --output_rtmp_url rtmp://localhost/processed/output \\
                 --device cuda --mode PLAYER_TRACKING
  
  # Real-time processing with resolution limit
  python main.py --stream_mode \\
                 --input_rtmp_url rtmp://localhost/live/input \\
                 --output_rtmp_url rtmp://localhost/processed/output \\
                 --device cuda --mode TEAM_CLASSIFICATION \\
                 --realtime --max_resolution 1280
        """
    )
    
    # File mode arguments
    parser.add_argument(
        '--source_video_path',
        type=str,
        help='Path to source video file (required for file mode)'
    )
    parser.add_argument(
        '--target_video_path',
        type=str,
        help='Path to output video file (required for file mode)'
    )
    
    # Streaming mode arguments
    parser.add_argument(
        '--stream_mode',
        action='store_true',
        help='Enable RTMP streaming mode'
    )
    parser.add_argument(
        '--input_rtmp_url',
        type=str,
        help='Input RTMP stream URL (e.g., rtmp://localhost/live/input)'
    )
    parser.add_argument(
        '--output_rtmp_url',
        type=str,
        help='Output RTMP stream URL (e.g., rtmp://localhost/processed/output)'
    )
    
    # Common arguments
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to run inference on (cpu, cuda, mps)'
    )
    parser.add_argument(
        '--mode',
        type=Mode,
        default=Mode.PLAYER_DETECTION,
        help='Processing mode'
    )
    
    # Performance arguments
    parser.add_argument(
        '--realtime',
        action='store_true',
        help='Use real-time single-pass processing (required for TEAM_CLASSIFICATION and RADAR in streaming mode)'
    )
    parser.add_argument(
        '--max_resolution',
        type=int,
        help='Maximum resolution (width or height) for input frames (e.g., 1280, 1920)'
    )
    
    args = parser.parse_args()
    
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode,
        stream_mode=args.stream_mode,
        input_rtmp_url=args.input_rtmp_url,
        output_rtmp_url=args.output_rtmp_url,
        realtime=args.realtime,
        max_resolution=args.max_resolution
    )
