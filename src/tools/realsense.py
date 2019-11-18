import time

import pyrealsense2 as rs
import numpy as np
import cv2
from . import loggingutil


class RealsenseSettings:
    def __init__(self, decimation: int = 0, frame_width: int = 640, frame_height: int = 480, fps: int = 30) -> object:
        self.decimation = decimation
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps


class RealsenseCamera:

    def __init__(self, settings: RealsenseSettings = None):
        self.logger = loggingutil.get_logger(__name__, do_file_logging=False)

        if settings is None:
            self.realsense_settings = RealsenseSettings()

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.camera_config = rs.config()

        self.camera_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        self.camera_config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)

        # Start streaming
        self.pipeline.start(self.camera_config)

        # Get stream profile and camera intrinsics
        profile = self.pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        # Processing blocks
        self.filter_decimate = rs.decimation_filter()
        self.filter_decimate.set_option(rs.option.filter_magnitude, 2 ** self.realsense_settings.decimation)
        self.filter_colorize = rs.colorizer()

        self.depth_intrinsics = None
        self.rgb_intrinsics = None
        self.get_frame()  # get a single frame to obtain valid intrinsics etc

    def get_depth_presets(self):
        depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
        preset_ids = depth_sensor.get_option_range(rs.option.visual_preset)
        presets = {}
        for i in range(int(preset_ids.max)):
            presets[depth_sensor.get_option_value_description(rs.option.visual_preset, i)] = i
        return presets

    def set_depth_preset(self, preset_id):
        depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
        self.logger.info("Setting  preset ""{}"" for depth sensor!".format(
            depth_sensor.get_option_value_description(rs.option.visual_preset, preset_id)))
        depth_sensor.set_option(rs.option.visual_preset, preset_id)
        self.pipeline.stop()
        self.pipeline.start(self.camera_config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame_raw = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = self.filter_decimate.process(depth_frame_raw)

        self.depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return time.time(), depth_image, color_image

    def get_current_intrinsics(self):
        self.logger.info("Intrinsics requested: {}".format(self.depth_intrinsics))
        return self.depth_intrinsics
