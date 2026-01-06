"""
RTMP Stream Processor for real-time video processing with hardware acceleration.

This module provides functionality to read from RTMP input streams, process frames
in real-time, and publish the results to RTMP output streams using hardware encoding
on NVIDIA Jetson devices.
"""

import logging
import subprocess
import time
from queue import Queue, Full, Empty
from threading import Thread, Event
from typing import Optional, Tuple, Callable
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RTMPStreamProcessor:
    """
    Handles RTMP input/output streaming with frame buffering and hardware encoding.
    
    This class manages:
    - Reading frames from RTMP input streams via OpenCV
    - Frame queue management for input/output buffering
    - Writing frames to RTMP output using FFmpeg with h264_nvenc hardware encoding
    - Performance monitoring (FPS, latency tracking)
    - Graceful error handling and reconnection logic
    """
    
    def __init__(
        self,
        input_rtmp_url: str,
        output_rtmp_url: str,
        max_queue_size: int = 30,
        reconnect_delay: int = 5,
        fps: int = 30,
        enable_hardware_encoding: bool = True
    ):
        """
        Initialize the RTMP Stream Processor.
        
        Args:
            input_rtmp_url: RTMP URL to read input stream from
            output_rtmp_url: RTMP URL to write output stream to
            max_queue_size: Maximum size of input/output frame queues
            reconnect_delay: Delay in seconds before reconnection attempts
            fps: Frames per second for output stream
            enable_hardware_encoding: Use h264_nvenc for hardware encoding (Jetson/NVIDIA GPU)
        """
        self.input_rtmp_url = input_rtmp_url
        self.output_rtmp_url = output_rtmp_url
        self.max_queue_size = max_queue_size
        self.reconnect_delay = reconnect_delay
        self.fps = fps
        self.enable_hardware_encoding = enable_hardware_encoding
        
        # Frame queues
        self.input_queue: Queue = Queue(maxsize=max_queue_size)
        self.output_queue: Queue = Queue(maxsize=max_queue_size)
        
        # Thread control
        self.stop_event = Event()
        self.input_thread: Optional[Thread] = None
        self.output_thread: Optional[Thread] = None
        
        # Stream properties
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        
        # Performance monitoring
        self.frames_read = 0
        self.frames_written = 0
        self.last_fps_time = time.time()
        self.input_fps = 0.0
        self.output_fps = 0.0
        self.latency_ms = 0.0
        
    def _connect_input_stream(self) -> bool:
        """
        Connect to the input RTMP stream.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to input stream: {self.input_rtmp_url}")
            self.cap = cv2.VideoCapture(self.input_rtmp_url)
            
            if not self.cap.isOpened():
                logger.error("Failed to open input stream")
                return False
            
            # Read first frame to get stream properties
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read first frame")
                return False
            
            self.height, self.width = frame.shape[:2]
            logger.info(f"Input stream connected: {self.width}x{self.height}")
            
            # Put first frame in queue
            try:
                self.input_queue.put((frame, time.time()), timeout=1)
            except Full:
                logger.warning("Input queue full, dropping frame")
            
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to input stream: {e}")
            return False
    
    def _connect_output_stream(self) -> bool:
        """
        Connect to the output RTMP stream using FFmpeg.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if self.width is None or self.height is None:
                logger.error("Cannot connect output stream: width/height not set")
                return False
            
            logger.info(f"Connecting to output stream: {self.output_rtmp_url}")
            
            # Build FFmpeg command
            encoder = 'h264_nvenc' if self.enable_hardware_encoding else 'libx264'
            preset = 'fast' if self.enable_hardware_encoding else 'ultrafast'
            
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{self.width}x{self.height}',
                '-r', str(self.fps),
                '-i', '-',  # Read from stdin
                '-c:v', encoder,
                '-preset', preset,
                '-b:v', '2M',
                '-g', str(self.fps * 2),  # GOP size
                '-f', 'flv',
                self.output_rtmp_url
            ]
            
            logger.info(f"Starting FFmpeg with command: {' '.join(ffmpeg_cmd)}")
            
            # Start FFmpeg process
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            
            logger.info("Output stream connected")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to output stream: {e}")
            return False
    
    def _input_thread_func(self):
        """Thread function for reading frames from input stream."""
        consecutive_failures = 0
        max_consecutive_failures = 30
        
        while not self.stop_event.is_set():
            # Ensure input stream is connected
            if self.cap is None or not self.cap.isOpened():
                if not self._connect_input_stream():
                    logger.warning(f"Reconnecting in {self.reconnect_delay} seconds...")
                    time.sleep(self.reconnect_delay)
                    continue
                consecutive_failures = 0
            
            try:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    consecutive_failures += 1
                    logger.warning(f"Failed to read frame ({consecutive_failures}/{max_consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("Too many consecutive failures, reconnecting...")
                        if self.cap:
                            self.cap.release()
                        self.cap = None
                        consecutive_failures = 0
                    continue
                
                consecutive_failures = 0
                
                # Add frame to queue with timestamp
                try:
                    self.input_queue.put((frame, time.time()), timeout=0.1)
                    self.frames_read += 1
                except Full:
                    logger.debug("Input queue full, dropping frame")
                
            except Exception as e:
                logger.error(f"Error in input thread: {e}")
                time.sleep(0.1)
        
        # Cleanup
        if self.cap:
            self.cap.release()
            logger.info("Input stream released")
    
    def _output_thread_func(self):
        """Thread function for writing frames to output stream."""
        consecutive_failures = 0
        max_consecutive_failures = 30
        
        while not self.stop_event.is_set():
            # Ensure output stream is connected
            if self.ffmpeg_process is None:
                if not self._connect_output_stream():
                    logger.warning(f"Reconnecting output in {self.reconnect_delay} seconds...")
                    time.sleep(self.reconnect_delay)
                    continue
                consecutive_failures = 0
            
            try:
                # Get frame from queue
                try:
                    frame, timestamp = self.output_queue.get(timeout=1)
                except Empty:
                    continue
                
                # Write frame to FFmpeg stdin
                try:
                    if self.ffmpeg_process and self.ffmpeg_process.stdin:
                        self.ffmpeg_process.stdin.write(frame.tobytes())
                        self.frames_written += 1
                        consecutive_failures = 0
                        
                        # Calculate latency
                        self.latency_ms = (time.time() - timestamp) * 1000
                    else:
                        consecutive_failures += 1
                        
                except (BrokenPipeError, IOError) as e:
                    consecutive_failures += 1
                    logger.warning(f"Error writing frame ({consecutive_failures}/{max_consecutive_failures}): {e}")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("Too many consecutive failures, reconnecting...")
                        if self.ffmpeg_process:
                            self.ffmpeg_process.stdin.close()
                            self.ffmpeg_process.wait()
                        self.ffmpeg_process = None
                        consecutive_failures = 0
                
            except Exception as e:
                logger.error(f"Error in output thread: {e}")
                time.sleep(0.1)
        
        # Cleanup
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.wait(timeout=5)
            except Exception as e:
                logger.error(f"Error closing FFmpeg process: {e}")
                self.ffmpeg_process.kill()
            logger.info("Output stream released")
    
    def start(self):
        """Start the input and output streaming threads."""
        logger.info("Starting RTMP Stream Processor...")
        
        self.stop_event.clear()
        
        # Start input thread
        self.input_thread = Thread(target=self._input_thread_func, daemon=True)
        self.input_thread.start()
        
        # Wait for stream properties to be initialized
        timeout = 30  # seconds
        start_time = time.time()
        while self.width is None and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        if self.width is None:
            raise RuntimeError("Failed to initialize stream properties")
        
        # Start output thread
        self.output_thread = Thread(target=self._output_thread_func, daemon=True)
        self.output_thread.start()
        
        logger.info("RTMP Stream Processor started")
    
    def stop(self):
        """Stop the streaming threads and cleanup resources."""
        logger.info("Stopping RTMP Stream Processor...")
        
        self.stop_event.set()
        
        # Wait for threads to finish
        if self.input_thread:
            self.input_thread.join(timeout=5)
        if self.output_thread:
            self.output_thread.join(timeout=5)
        
        logger.info("RTMP Stream Processor stopped")
    
    def read_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Read a frame from the input queue.
        
        Args:
            timeout: Maximum time to wait for a frame
            
        Returns:
            Frame as numpy array, or None if timeout or stopped
        """
        try:
            frame, timestamp = self.input_queue.get(timeout=timeout)
            return frame
        except Empty:
            return None
    
    def write_frame(self, frame: np.ndarray):
        """
        Write a frame to the output queue.
        
        Args:
            frame: Frame to write as numpy array
        """
        try:
            self.output_queue.put((frame, time.time()), timeout=0.1)
        except Full:
            logger.debug("Output queue full, dropping frame")
    
    def get_stream_properties(self) -> Tuple[int, int, int]:
        """
        Get stream properties.
        
        Returns:
            Tuple of (width, height, fps)
        """
        return self.width or 0, self.height or 0, self.fps
    
    def get_performance_stats(self) -> dict:
        """
        Get current performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed >= 1.0:  # Update every second
            self.input_fps = self.frames_read / elapsed
            self.output_fps = self.frames_written / elapsed
            self.frames_read = 0
            self.frames_written = 0
            self.last_fps_time = current_time
        
        return {
            'input_fps': self.input_fps,
            'output_fps': self.output_fps,
            'latency_ms': self.latency_ms,
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize()
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
