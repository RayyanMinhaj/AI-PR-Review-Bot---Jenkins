import logging
import multiprocessing as mp
import os
import subprocess as sp
import sys




from frigate.config import FrigateConfig  # noqa: E402
from frigate.motion import MotionDetector  # noqa: E402
from frigate.object_detection import LocalObjectDetector  # noqa: E402
from frigate.object_processing import CameraState  # noqa: E402
from frigate.track.centroid_tracker import CentroidTracker  # noqa: E402
from frigate.util import (  # noqa: E402
    EventsPerSecond,
    SharedMemoryFrameManager,
    draw_box_with_label,
)
from frigate.video import (  # noqa: E402
    capture_frames,
    process_frames,
    start_or_restart_ffmpeg,
)

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


def get_frame_shape(source):
    ffprobe_cmd = [
        "ffprobe",
        "-v",
        "panic",
        "-show_error",
        "-show_streams",
        "-of",
        "json",
        source,
    ]
    p = sp.run(ffprobe_cmd, capture_output=True)
    info = json.loads(p.stdout)

    video_info = [s for s in info["streams"] if s["codec_type"] == "video"][0]

    if video_info["height"] != 0 and video_info["width"] != 0:
        return (video_info["height"], video_info["width"], 3)

    # fallback to using opencv if ffprobe didn't succeed
    video = cv2.VideoCapture(source)
    ret, frame = video.read()
    frame_shape = frame.shape
    video.release()
    return frame_shape


class ProcessClip:
    def __init__(self, clip_path, frame_shape, config: FrigateConfig):
        self.clip_path = clip_path
        self.camera_name = "camera"
        self.config = config
        self.camera_config = self.config.cameras["camera"]
        self.frame_shape = self.camera_config.frame_shape
        self.ffmpeg_cmd = [
            c["cmd"] for c in self.camera_config.ffmpeg_cmds if "detect" in c["roles"]
        ][0]
        self.frame_manager = SharedMemoryFrameManager()
        self.frame_queue = mp.Queue()
        self.detected_objects_queue = mp.Queue()
        self.camera_state = CameraState(self.camera_name, config, self.frame_manager)

    def load_frames(self):
        fps = EventsPerSecond()
        skipped_fps = EventsPerSecond()
        current_frame = mp.Value("d", 0.0)
        frame_size = (
            self.camera_config.frame_shape_yuv[0]
            * self.camera_config.frame_shape_yuv[1]
        )
        ffmpeg_process = start_or_restart_ffmpeg(
            self.ffmpeg_cmd, logger, sp.DEVNULL, frame_size
        )
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

    def process_frames(
        self, object_detector, objects_to_track=["person"], object_filters={}
    ):
        mask = np.zeros((self.frame_shape[0], self.frame_shape[1], 1), np.uint8)
        mask[:] = 255
        motion_detector = MotionDetector(self.frame_shape, self.camera_config.motion)
        motion_detector.save_images = False

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

    def stats(self, debug_path=None):
        total_regions = 0
        total_motion_boxes = 0
        object_ids = set()
        total_frames = 0

        while not self.detected_objects_queue.empty():
            (
                camera_name,
                frame_time,
                current_tracked_objects,
                motion_boxes,
                regions,
            ) = self.detected_objects_queue.get()

            if debug_path:
                self.save_debug_frame(
                    debug_path, frame_time, current_tracked_objects.values()
                )

            self.camera_state.update(
                frame_time, current_tracked_objects, motion_boxes, regions
            )
            total_regions += len(regions)
            total_motion_boxes += len(motion_boxes)
            top_score = 0
            for id, obj in self.camera_state.tracked_objects.items():
                if not obj.false_positive:
                    object_ids.add(id)
                    if obj.top_score > top_score:
                        top_score = obj.top_score

            total_frames += 1
