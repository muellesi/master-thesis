import cv2
import numpy as np

def render_bb(img, bb, label=None):
    """
        Renders a bounding box on the given image
        :param label: text to be put on the bounding box
        :param img: numpy array, image
        :param bb: bounding box in the format [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    """
    assert img.shape[2] == 3
    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 2)
    if label is not None:
        # see https://gist.github.com/aplz/fd34707deffb208f367808aade7e5d5c#file-draw_text_with_background_opencv-py
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height) = cv2.getTextSize(str(label), font, fontScale=1, thickness=1)[0]
        text_offset_x = bb[0] + 2
        text_offset_y = bb[1] - 4
        box_coords = ((text_offset_x - 4, text_offset_y + 2), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
        cv2.rectangle(img, box_coords[0], box_coords[1], (255, 0, 0), cv2.FILLED)
        cv2.putText(img, str(label), (text_offset_x, text_offset_y), font, fontScale=1, color=(255, 255, 255), thickness=1)
    return img
