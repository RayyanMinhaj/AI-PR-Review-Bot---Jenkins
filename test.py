import csv
import json
import logging
import multiprocessing as mp
import os
import subprocess as sp
import sys

import click
import cv2
import numpy as np

sys.path.append("/workspace/frigate")

from frigate.config import FrigateConfig
from frigate.motion import MotionDetector
from frigate.object_detection import LocalObjectDetector
from frigate.object_processing import CameraState
from frigate.track.centroid_tracker import CentroidTracker
from frigate.util import (
    EventsPerSecond,
    SharedMemoryFrameManager,
    draw_box_with_label,
)
from frigate.video import (
    capture_frames,
    process_frames,
    start_or_restart_ffmpeg,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_frame_shape(source):
    """
    Gets the shape of the video frame using ffprobe, falling back to OpenCV if necessary.
    """
    try:
        ffprobe_cmd = [
            "ffprobe", "-v", "panic", "-show_error", "-show_streams", "-of", "json", source
        ]
        process = sp.run(ffprobe_cmd, capture_output=True)
        info = json.loads(process.stdout)
        video_info = next(s for s in info["streams"] if s["codec_type"] == "video")
        return (video_info["height"], video_info["width"], 3)
    except (KeyError, IndexError):
        # Fallback to OpenCV if ffprobe fails
        video = cv2.VideoCapture(source)
        ret, frame = video.read()
        if not ret:
            raise ValueError("Unable to capture frames from video.")
        frame_shape = frame.shape
        video.release()
        return frame_shape


class ProcessClip:
    def __init__(self, clip_path, frame_shape, config: FrigateConfig):
        self.clip_path = clip_path
        self.camera_name = "camera"
        self.config = config
        self.camera_config = self.config.cameras["camera"]
        self.frame_shape = frame_shape
        self.frame_manager = SharedMemoryFrameManager()
        self.frame_queue = mp.Queue()
        self.detected_objects_queue = mp.Queue()
        self.camera_state = CameraState(self.camera_name, config, self.frame_manager)

        # Get FFmpeg command for detection role
        self.ffmpeg_cmd = next(
            (c["cmd"] for c in self.camera_config.ffmpeg_cmds if "detect" in c["roles"]),
            None,
        )

    def load_frames(self):
        """
        Loads frames from the clip using FFmpeg.
        """
        fps = EventsPerSecond()
        skipped_fps = EventsPerSecond()
        current_frame = mp.Value("d", 0.0)
        frame_size = (
            self.camera_config.frame_shape_yuv[0] * self.camera_config.frame_shape_yuv[1]
        )

        ffmpeg_process = start_or_restart_ffmpeg(self.ffmpeg_cmd, logger, sp.DEVNULL, frame_size)
        capture_frames(
            ffmpeg_process,
            self.camera_name,
            self.camera_config.frame_shape_yuv,
            self.frame_manager,
            self.frame_queue,
            fps,
            skipped_fps,
            current_frame,
        )
        ffmpeg_process.wait()
        ffmpeg_process.communicate()

    def process_frames(self, object_detector, objects_to_track=["person"], object_filters={}):
        """
        Processes the video frames to detect and track objects.
        """
        mask = np.full((self.frame_shape[0], self.frame_shape[1], 1), 255, dtype=np.uint8)
        motion_detector = MotionDetector(self.frame_shape, self.camera_config.motion)
        object_tracker = CentroidTracker(self.camera_config.detect)

        process_info = {
            "process_fps": mp.Value("d", 0.0),
            "detection_fps": mp.Value("d", 0.0),
            "detection_frame": mp.Value("d", 0.0),
        }

        detection_enabled = mp.Value("d", 1)
        motion_enabled = mp.Value("d", True)
        stop_event = mp.Event()

        process_frames(
            self.camera_name,
            self.frame_queue,
            self.frame_shape,
            self.config.model,
            self.camera_config.detect,
            self.frame_manager,
            motion_detector,
            object_detector,
            object_tracker,
            self.detected_objects_queue,
            process_info,
            objects_to_track,
            object_filters,
            detection_enabled,
            motion_enabled,
            stop_event,
            exit_on_empty=True,
        )


if __name__ == "__main__":
    process()
