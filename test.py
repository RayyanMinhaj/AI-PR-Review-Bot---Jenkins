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

    def stats(self, debug_path=None):
        """
        Gathers stats about object detections and motion from the video.
        """
        total_regions = 0
        total_motion_boxes = 0
        object_ids = set()
        total_frames = 0
        top_score = 0

        while not self.detected_objects_queue.empty():
            (
                camera_name,
                frame_time,
                current_tracked_objects,
                motion_boxes,
                regions,
            ) = self.detected_objects_queue.get()

            if debug_path:
                self.save_debug_frame(debug_path, frame_time, current_tracked_objects.values())

            self.camera_state.update(
                frame_time, current_tracked_objects, motion_boxes, regions
            )

            total_regions += len(regions)
            total_motion_boxes += len(motion_boxes)

            for id, obj in self.camera_state.tracked_objects.items():
                if not obj.false_positive:
                    object_ids.add(id)
                    if obj.top_score > top_score:
                        top_score = obj.top_score

            total_frames += 1
            self.frame_manager.delete(self.camera_state.previous_frame_id)

        return {
            "total_regions": total_regions,
            "total_motion_boxes": total_motion_boxes,
            "true_positive_objects": len(object_ids),
            "total_frames": total_frames,
            "top_score": top_score,
        }

    def save_debug_frame(self, debug_path, frame_time, tracked_objects):
        """
        Saves debug frames with bounding boxes to the specified path.
        """
        current_frame = cv2.cvtColor(
            self.frame_manager.get(
                f"{self.camera_name}{frame_time}", self.camera_config.frame_shape_yuv
            ),
            cv2.COLOR_YUV2BGR_I420,
        )

        for obj in tracked_objects:
            color, thickness = ((255, 255, 0), 2) if obj["frame_time"] == frame_time else ((255, 0, 0), 1)
            draw_box_with_label(
                current_frame,
                *obj["box"],
                obj["id"],
                f"{int(obj['score']*100)}% {int(obj['area'])}",
                thickness=thickness,
                color=color,
            )
            region = obj["region"]
            draw_box_with_label(current_frame, *region, "region", "", thickness=1, color=(0, 255, 0))

        cv2.imwrite(
            f"{os.path.join(debug_path, os.path.basename(self.clip_path))}.{int(frame_time*1000000)}.jpg",
            current_frame,
        )


@click.command()
@click.option("-p", "--path", required=True, help="Path to clip or directory to test.")
@click.option("-l", "--label", default="person", help="Label name to detect.")
@click.option("-o", "--output", default=None, help="File to save csv of data.")
@click.option("--debug-path", default=None, help="Path to output frames for debugging.")
def process(path, label, output, debug_path):
    """
    Command line entry point to process video clips and output object detection statistics.
    """
    clips = []
    if os.path.isdir(path):
        clips = [os.path.join(path, file) for file in sorted(os.listdir(path))]
    elif os.path.isfile(path):
        clips.append(path)

    json_config = {
        "mqtt": {"enabled": False},
        "detectors": {"coral": {"type": "edgetpu", "device": "usb"}},
        "cameras": {
            "camera": {
                "ffmpeg": {
                    "inputs": [
                        {
                            "path": "path.mp4",
                            "global_args": "-hide_banner",
                            "input_args": "-loglevel info",
                            "roles": ["detect"],
                        }
                    ]
                },
                "record": {"enabled": False},
            }
        },
    }

    object_detector = LocalObjectDetector(labels="/labelmap.txt")
    results = []

    for clip in clips:
        logger.info(f"Processing clip: {clip}")
        frame_shape = get_frame_shape(clip)
        json_config["cameras"]["camera"]["detect"] = {"height": frame_shape[0], "width": frame_shape[1]}
        json_config["cameras"]["camera"]["ffmpeg"]["inputs"][0]["path"] = clip

        frigate_config = FrigateConfig(**json_config)
        process_clip = ProcessClip(clip, frame_shape, frigate_config)
        process_clip.load_frames()
        process_clip.process_frames(object_detector, objects_to_track=[label])

        results.append((clip, process_clip.stats(debug_path)))

    positive_count = sum(1 for _, stats in results if stats["true_positive_objects"] > 0)
    logger.info(f"Objects detected in {positive_count}/{len(results)} clips ({(positive_count/len(results)*100):.2f}%).")

    if output:
        with open(output, "w", newline="") as data_file:
            csv_writer = csv.writer(data_file)
            header_written = False

            for clip, stats in results:
                if not header_written:
                    csv_writer.writerow(["file"] + list(stats.keys()))
                    header_written = True
                csv_writer.writerow([clip] + list(stats.values()))


if __name__ == "__main__":
    process()
