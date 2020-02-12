import time
import os
import json
import pyrealsense2 as rs
import numpy as np
import cv2
from . import loggingutil
import tools
import time



# partly taken from https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/python-rs400-advanced-mode-example.py#L80
# and https://github.com/IntelRealSense/librealsense/issues/856
__DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07", "0B3A"]


class RealsenseCamera:

    def __init__(self, settings=None):
        self.logger = loggingutil.get_logger(__name__, do_file_logging=False)

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.camera_config = rs.config()

        if settings is not None:
            settings_json = settings['file']
            if os.path.exists(settings_json):
                json_obj = json.load(open(settings_json))
                json_string = str(json_obj).replace("'", '\"')
                print("Configuration " + settings_json + " loaded")

                self.camera_config.enable_stream(rs.stream.depth, int(json_obj['stream-width']), int(json_obj['stream-height']), rs.format.z16, 30)
                self.camera_config.enable_stream(rs.stream.color, int(json_obj['stream-width']), int(json_obj['stream-height']), rs.format.bgr8, 30)

                # Start streaming
                self.cfg = self.pipeline.start(self.camera_config)
                self.dev = self.cfg.get_device()

                self.advnc_mode = rs.rs400_advanced_mode(self.dev)
                print("Advanced mode is", "enabled" if self.advnc_mode.is_enabled() else "disabled")

                # Loop until we successfully enable advanced mode
                while not self.advnc_mode.is_enabled():
                    print("Trying to enable advanced mode...")
                    self.advnc_mode.toggle_advanced_mode(True)
                    time.sleep(5)
                    # The 'dev' object will become invalid and we need to initialize it again
                    self.dev = self.__find_device_that_supports_advanced_mode()
                    self.advnc_mode = rs.rs400_advanced_mode(self.dev)
                    print("Advanced mode is", "enabled" if self.advnc_mode.is_enabled() else "disabled")

                self.advnc_mode.load_json(json_string)
            else:
                # Start streaming
                self.cfg = self.pipeline.start(self.camera_config)
                self.dev = self.cfg.get_device()
                self.logger.error("Given config file path {} is not valid!".format(settings_json))

        # Get stream profile and camera intrinsics
        profile = self.pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        # Processing blocks
        self.filter_decimate = rs.decimation_filter()
        self.filter_decimate.set_option(rs.option.filter_magnitude, 2 ** 0)
        self.filter_spatial = rs.spatial_filter(0.5, 20.0, 2.0, 0)
        self.filter_temporal = rs.temporal_filter(0.4, 20, 1)
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        self.filter_colorize = rs.colorizer()

        self.depth_intrinsics = None
        self.rgb_intrinsics = None
        self.get_frame()  # get a single frame to obtain valid intrinsics etc


    def __find_device_that_supports_advanced_mode(self):
        global __DS5_product_ids
        ctx = rs.context()
        ds5_dev = rs.device()
        devices = ctx.query_devices()
        for dev in devices:
            if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in __DS5_product_ids:
                if dev.supports(rs.camera_info.name):
                    print("Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
                return dev
        raise Exception("No device that supports advanced mode was found")


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
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        #depth_frame = self.filter_decimate.process(depth_frame_raw)
        depth_frame = self.depth_to_disparity.process(depth_frame)
        depth_frame = self.filter_spatial.process(depth_frame)
        #depth_frame = self.filter_temporal.process(depth_frame)
        depth_frame = self.disparity_to_depth.process(depth_frame)

        self.depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return time.time(), depth_image, color_image


    def get_current_intrinsics(self):
        # self.logger.debug("Intrinsics requested: {}".format(self.depth_intrinsics))

        return np.array([[self.depth_intrinsics.fx, 0.0, self.depth_intrinsics.ppx],
                         [0.0, self.depth_intrinsics.fy, self.depth_intrinsics.ppy],
                         [0.0, 0.0, 1.0]])


    def autocycle_stream(self, cmap, num_cycles=-1):
        cv2.namedWindow("autocycleStream")
        loop_end = num_cycles if num_cycles >= 0 else 1
        i = 0
        while i < loop_end:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            # color_frame = frames.get_color_frame()
            if not depth_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = tools.colorize(depth_image, vmin=0, vmax=4000, cmap=cmap).numpy()
            cv2.imshow('autocycleStream', depth_colormap)
            if num_cycles >= 0:
                i = i + 1
            cv2.waitKey(1)
        cv2.destroyAllWindows()
