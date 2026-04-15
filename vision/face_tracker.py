"""Face detection and tracking using MediaPipe.

Provides real-time face detection with smoothing for stable head tracking.
"""

import time
import threading
from typing import Optional, Tuple
from collections import deque

import cv2


class FaceTracker:
    """Face detection and position tracking.
    
    Uses MediaPipe Face Detection for efficient CPU-based face tracking.
    Provides smoothed face center coordinates for robot head control.
    
    Args:
        model_selection: 0 for short-range (2m), 1 for long-range (5m)
        min_detection_confidence: Detection threshold (0.0-1.0)
        smooth_factor: EMA smoothing factor (0.0-1.0, higher = more responsive)
    """
    
    def __init__(
        self,
        model_selection: int = 0,
        min_detection_confidence: float = 0.5,
        smooth_factor: float = 0.25,  # High smoothness
        multi_face_strategy: str = "largest"  # "largest", "center", "leftmost"
    ):
        self.model_selection = model_selection
        self.min_detection_confidence = min_detection_confidence
        self.smooth_factor = smooth_factor
        self.multi_face_strategy = multi_face_strategy
        
        # MediaPipe will be imported on first use to avoid startup overhead
        self._face_detection = None
        self._mp_drawing = None

        # Fallback detector when mediapipe.solutions is unavailable in some builds
        self._fallback_detector = None
        self._using_fallback = False
        
        # Tracking state
        self._current_position: Optional[Tuple[int, int]] = None
        self._last_detection_time: float = 0.0
        self._detection_timeout: float = 1.0  # seconds
        
        # Smoothing
        self._ema_x: Optional[float] = None
        self._ema_y: Optional[float] = None
        
        # Statistics
        self._fps_history = deque(maxlen=30)
        self._last_frame_time: float = 0.0
        
    def _init_mediapipe(self):
        """Lazy initialization of MediaPipe.

        Some newer/bundled mediapipe builds ship tasks-only APIs and do not
        expose `mediapipe.solutions`. In that case, fall back to OpenCV
        Haar cascade so tracking still works.
        """
        if self._face_detection is not None or self._fallback_detector is not None:
            return

        try:
            import mediapipe as mp
            has_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection")
            if has_solutions:
                print(
                    f"      📦 Initializing MediaPipe FaceDetection "
                    f"(model={self.model_selection}, conf={self.min_detection_confidence})"
                )
                self._face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=self.model_selection,
                    min_detection_confidence=self.min_detection_confidence,
                )
                self._mp_drawing = getattr(mp.solutions, "drawing_utils", None)
                self._using_fallback = False
                print("      ✅ MediaPipe initialized")
                return
            print("      ⚠️ mediapipe.solutions not available, switching to OpenCV Haar face detector")
        except ImportError as e:
            print(f"      ⚠️ MediaPipe import failed: {e}. Falling back to OpenCV Haar detector")

        # Fallback path: OpenCV Haar cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._fallback_detector = cv2.CascadeClassifier(cascade_path)
        if self._fallback_detector.empty():
            raise RuntimeError(
                "Failed to initialize fallback face detector (OpenCV Haar cascade)."
            )
        self._using_fallback = True
    
    def detect(self, frame) -> Optional[Tuple[int, int, int, int]]:
        """Detect face in frame and return bounding box.
        
        Args:
            frame: OpenCV BGR image (numpy array)
            
        Returns:
            Tuple of (x, y, width, height) or None if no face detected
        """
        self._init_mediapipe()
        
        # Validate frame
        if frame is None or frame.size == 0:
            print("      ⚠️  Invalid frame (None or empty)")
            return None
        
        h, w = frame.shape[:2]
        if h < 100 or w < 100:
            print(f"      ⚠️  Frame too small: {w}x{h}")
            return None
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame (MediaPipe uses process(), not detect())
        if self._using_fallback and self._fallback_detector is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._fallback_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )
            if len(faces) == 0:
                return None
            if len(faces) == 1:
                x, y, fw, fh = faces[0]
                return (int(x), int(y), int(fw), int(fh))

            # Multiple faces: reuse strategy by adapting to a common shape
            class _BBox:
                def __init__(self, x0, y0, w0, h0):
                    self.xmin = x0 / frame_w
                    self.ymin = y0 / frame_h
                    self.width = w0 / frame_w
                    self.height = h0 / frame_h

            class _Loc:
                def __init__(self, bb):
                    self.relative_bounding_box = bb

            class _Det:
                def __init__(self, x0, y0, w0, h0):
                    self.location_data = _Loc(_BBox(x0, y0, w0, h0))

            frame_h, frame_w = frame.shape[:2]
            detections = [_Det(int(x), int(y), int(fw), int(fh)) for (x, y, fw, fh) in faces]
            detection = self._select_face(detections, frame_w, frame_h)
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * frame_w)
            y = int(bbox.ymin * frame_h)
            width = int(bbox.width * frame_w)
            height = int(bbox.height * frame_h)
            return (x, y, width, height)

        results = self._face_detection.process(rgb_frame)
        
        if not results or not results.detections:
            return None
        
        # Log multiple faces
        num_faces = len(results.detections)
        if num_faces > 1:
            print(f"      👥 {num_faces} faces detected, using '{self.multi_face_strategy}' strategy")
        
        # Select face based on strategy
        if num_faces == 1:
            detection = results.detections[0]
        else:
            # Multiple faces - apply selection strategy
            detection = self._select_face(results.detections, w, h)
        
        bbox = detection.location_data.relative_bounding_box
        
        # Convert relative coordinates to absolute pixels
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        return (x, y, width, height)
    
    def _select_face(self, detections, frame_w, frame_h):
        """Select which face to track when multiple detected.
        
        Strategies:
        - "largest": Biggest face (closest person)
        - "center": Face closest to image center (main subject)
        - "leftmost": Leftmost face (reading order)
        """
        if self.multi_face_strategy == "largest":
            # Select largest face by area
            largest = None
            max_area = 0
            for det in detections:
                bbox = det.location_data.relative_bounding_box
                area = bbox.width * bbox.height
                if area > max_area:
                    max_area = area
                    largest = det
            return largest
        
        elif self.multi_face_strategy == "center":
            # Select face closest to image center
            center_x, center_y = frame_w / 2, frame_h / 2
            closest = None
            min_dist = float('inf')
            for det in detections:
                bbox = det.location_data.relative_bounding_box
                face_cx = (bbox.xmin + bbox.width / 2) * frame_w
                face_cy = (bbox.ymin + bbox.height / 2) * frame_h
                dist = ((face_cx - center_x) ** 2 + (face_cy - center_y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    closest = det
            return closest
        
        else:  # "leftmost" or default
            # Select leftmost face
            leftmost = None
            min_x = float('inf')
            for det in detections:
                bbox = det.location_data.relative_bounding_box
                x = bbox.xmin * frame_w
                if x < min_x:
                    min_x = x
                    leftmost = det
            return leftmost
    
    def get_face_center(self, frame) -> Optional[Tuple[int, int]]:
        """Get smoothed face center coordinates.
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            Tuple of (x, y) pixel coordinates or None
        """
        bbox = self.detect(frame)
        
        if bbox is None:
            # Check if we should clear stale position
            if time.time() - self._last_detection_time > self._detection_timeout:
                self._current_position = None
                self._ema_x = None
                self._ema_y = None
            return self._current_position
        
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Apply EMA smoothing
        if self._ema_x is None:
            self._ema_x = center_x
            self._ema_y = center_y
        else:
            self._ema_x = (
                self.smooth_factor * center_x + 
                (1 - self.smooth_factor) * self._ema_x
            )
            self._ema_y = (
                self.smooth_factor * center_y + 
                (1 - self.smooth_factor) * self._ema_y
            )
        
        self._current_position = (int(self._ema_x), int(self._ema_y))
        self._last_detection_time = time.time()
        
        return self._current_position
    
    def get_position(self) -> Optional[Tuple[int, int]]:
        """Get last known face position without processing new frame.
        
        Returns:
            Last smoothed (x, y) or None if no recent detection
        """
        # Check for timeout
        if time.time() - self._last_detection_time > self._detection_timeout:
            self._current_position = None
        return self._current_position
    
    def is_face_detected(self) -> bool:
        """Check if face is currently tracked."""
        return self.get_position() is not None
    
    def update_fps(self):
        """Update FPS calculation. Call once per frame."""
        current_time = time.time()
        if self._last_frame_time > 0:
            fps = 1.0 / (current_time - self._last_frame_time)
            self._fps_history.append(fps)
        self._last_frame_time = current_time
    
    def get_fps(self) -> float:
        """Get average FPS over last 30 frames."""
        if not self._fps_history:
            return 0.0
        return sum(self._fps_history) / len(self._fps_history)
    
    def close(self):
        """Release resources."""
        if self._face_detection is not None:
            self._face_detection.close()


class FaceTrackerThread(threading.Thread):
    """Background thread for continuous face tracking.
    
    Runs face detection in a separate thread to avoid blocking
    the main chat loop.
    
    Args:
        camera: Callable that returns current frame
        tracker: FaceTracker instance
        callback: Optional callback(face_position) on each detection
    """
    
    def __init__(
        self,
        camera,
        tracker: FaceTracker,
        callback=None,
        fps_target: float = 15.0
    ):
        super().__init__(daemon=True)
        self.camera = camera
        self.tracker = tracker
        self.callback = callback
        self.fps_target = fps_target
        self._running = False
        self._frame_interval = 1.0 / fps_target
        
    def run(self):
        """Main tracking loop."""
        self._running = True
        
        while self._running:
            start_time = time.time()
            
            # Get frame from camera
            frame = self.camera()
            if frame is not None:
                # Track face
                pos = self.tracker.get_face_center(frame)
                self.tracker.update_fps()
                
                # Notify callback
                if self.callback and pos:
                    self.callback(pos)
            
            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def stop(self):
        """Stop the tracking thread."""
        self._running = False
