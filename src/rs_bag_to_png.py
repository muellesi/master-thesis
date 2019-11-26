##########################################################################################################################
##                                          Read bag from file                                                          ##
## Source: https://github.com/IntelRealSense/librealsense/blob/development/wrappers/python/examples/read_bag_example.py ##
##                                          Modified by Simon Müller                                                    ##
##########################################################################################################################


# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path



# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream resolution, fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
# Parse the command line arguments to an object
args = parser.parse_args()
# Safety if no parameter have been given
if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()
# Check if the given file have bag extension
if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()
try:

    savepath = os.path.abspath(os.path.join(args.input, os.pardir))

    # Create pipeline
    print("Creating pipeline")
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()
    # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
    print("Loading bag...")
    rs.config.enable_device_from_file(config, args.input)
    # Configure the pipeline to stream the depth stream
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming from file
    print("Starting stream...")
    pipeline.start(config)

    # Create opencv window to render image in
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)

    # Create colorizer object
    colorizer = rs.colorizer()
    thresholder = rs.threshold_filter(0.0, 0.6)

    base_frame_id = -1
    frame_id = 0
    stop = False

    # Streaming loop
    while True:
        # Get frameset of depth
        frames = pipeline.wait_for_frames()

        # Get depth frame
        depth_frame = frames.get_depth_frame()
        depth_frame = thresholder.process(depth_frame)

        if base_frame_id == -1:
            base_frame_id = depth_frame.get_frame_number()
            frame_id = depth_frame.get_frame_number() - base_frame_id
        else:
            frame_id = depth_frame.get_frame_number() - base_frame_id
            if frame_id == 0: break
        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        # Render image in opencv window
        cv2.imshow("Depth Stream", depth_color_image)

        print("Saving frame #{:07d} as {}".format(frame_id, os.path.join(savepath, "depth_{:07d}.png".format(frame_id))))
        img = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        cv2.imwrite(os.path.join(savepath, "depth_{:07d}.png".format(frame_id)), img)

        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break


finally:
    pass
