import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2



def joint_angle_from_joint_positions(joint1, joint2, joint3):
    dir12 = joint2 - joint1
    dir23 = joint3 - joint2

    angle_joint_2 = np.arccos(np.dot(dir12, dir23) / np.linalg.norm(dir12) * np.linalg.norm(dir23))
    return angle_joint_2


# Display utilities
def visualize_joints_2d(canvas, joints, joint_idxs=True, joint_names=None, links=None, alpha=1):
    """Draw 2d skeleton on matplotlib axis
    Based on https://github.com/guiggh/hand_pose_action/blob/master/load_example.py#L49
    """
    if links is None:
        links = [(0, 1, 6, 7, 8), (0, 2, 9, 10, 11), (0, 3, 12, 13, 14),
                 (0, 4, 15, 16, 17), (0, 5, 18, 19, 20)]

    if isinstance(canvas, matplotlib.axes.SubplotBase):
        ax = canvas
        # Scatter hand joints on image
        x = joints[:, 0]
        y = joints[:, 1]
        ax.scatter(x, y, 1, 'r')

        # Add idx labels to joints
        for row_idx, row in enumerate(joints):
            if joint_idxs:
                if joint_names:
                    plt.annotate(str(joint_names[row_idx]), (row[0], row[1]))
                else:
                    plt.annotate(str(row_idx), (row[0], row[1]))

        __draw2djoints_matplotlib(ax, joints, links, alpha=alpha)
        return None

    elif isinstance(canvas, np.ndarray):
        img = canvas

        for joint in joints:
            joint = joint.astype(np.int)
            img = cv2.circle(img, tuple(joint), 1, (255, 0, 0), 1)
        img = __draw2djoints_cv(img, joints, links, alpha=alpha)
        return img


def __draw2djoints_matplotlib(ax, annots, links, alpha=1):
    """
    Draw segments, one color per link
    https://github.com/guiggh/hand_pose_action/blob/master/load_example.py#L66
    """
    colors = ['r', 'm', 'b', 'c', 'g']

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            __draw2dseg_matplotlib(
                    ax,
                    annots,
                    finger_links[idx],
                    finger_links[idx + 1],
                    c=colors[finger_idx],
                    alpha=alpha)


def __draw2dseg_matplotlib(ax, annot, idx1, idx2, c='r', alpha=1):
    """Draw segment of given color
    https://github.com/guiggh/hand_pose_action/blob/master/load_example.py#L81
    """
    ax.plot(
            [annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]],
            c=c,
            alpha=alpha)


def __draw2dseg_cv(img, joints, idx1, idx2, c='r', alpha=1):
    cmap = {
            'r': (255, 0, 0),
            'm': (255, 0, 255),
            'b': (0, 0, 240),
            'c': (0, 255, 255),
            'g': (0, 240, 0)
            }
    img = cv2.line(img, tuple(joints[idx1].astype(np.int)), tuple(joints[idx2].astype(np.int)), cmap[c])
    return img


def __draw2djoints_cv(img, joints, links, alpha=1):
    """
    Draw segments, one color per link
    https://github.com/guiggh/hand_pose_action/blob/master/load_example.py#L66
    """
    colors = ['r', 'm', 'b', 'c', 'g']

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            img = __draw2dseg_cv(
                    img,
                    joints,
                    finger_links[idx],
                    finger_links[idx + 1],
                    c=colors[finger_idx],
                    alpha=alpha)
    return img


def project_world_to_cam(skel, cam_extrinsics):
    skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
    skel_camcoords = cam_extrinsics.dot(
            skel_hom.transpose()).transpose()[:, :3].astype(np.float32)
    return skel_camcoords


def project_2d(skel, cam_intrinsics):
    """"
        Projects a given skeleton from 3d to 2d, using the given camera intrinsics
        Basic code source: https://github.com/guiggh/hand_pose_action/blob/master/load_example.py#L147
        :param skel: Ax3 coordinate matrix of the skeleton with A joints
        :param cam_intrinsics: 4x3 camera intrinsics matrix
    """
    skel_hom2d = np.array(cam_intrinsics).dot(skel.transpose()).transpose()
    skel_proj = (skel_hom2d / skel_hom2d[:, 2:])[:, :2]
    return skel_proj
