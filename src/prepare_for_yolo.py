import tensorflow as tf
import numpy as np
from datasets.tfrecord_helper import depth_and_skel
import tools
import cv2
import json
import os
import shutil
from datasets.serialized_dataset import SerializedDataset

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree



class PascalVocWriter:
    """
    Source: https://github.com/manhcuogntin4/Label-Annotation-VOC-Pascal/blob/aef69b078443ad0baab4d189ddde3ab165ae9b23/libs/pascal_voc_io.py
    """


    def __init__(self, foldername, filename, imgSize, databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath


    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8', )
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True)


    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None or \
                len(self.boxlist) <= 0:
            return None

        top = Element('annotation')
        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        localImgPath = SubElement(top, 'path')
        localImgPath.text = self.localImgPath

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top


    def addBndBox(self, xmin, ymin, xmax, ymax, name):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        self.boxlist.append(bndbox)


    def appendObjects(self, top):
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            name.text = each_object['name']
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = "0"
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])


    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = open(self.filename + '.xml', 'w')
        else:
            out_file = open(targetFile, 'w')

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


def bounding_box(skel, margin=0.0):
    min_x, min_y = np.min(skel, axis=0)
    max_x, max_y = np.max(skel, axis=0)
    return np.array([int(min_x - margin), int(min_y - margin), int(max_x + margin), int(max_y + margin)])


def prepare(ds_provider: SerializedDataset, output_path):
    ds_types = ['train', 'validation', 'test']
    intr = ds_provider.camera_intrinsics

    #win_name = "ds_view"
    #cv2.namedWindow(win_name)
    for ds_type in ds_types:
        img_path = os.path.join(output_path, "{}_img".format(ds_type))
        annot_path = os.path.join(output_path, "{}_annot".format(ds_type))

        if os.path.exists(img_path):
            shutil.rmtree(img_path)
        if os.path.exists(annot_path):
            shutil.rmtree(annot_path)

        os.makedirs(img_path)
        os.makedirs(annot_path)

        dataset = ds_provider.get_data(ds_type)
        dataset = dataset.map(depth_and_skel, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        img_idx = 0

        for img, skel in dataset:
            skel = skel.numpy()
            skel = skel.reshape((21, 3))
            skel = tools.project_2d(skel, intr)
            img = img.numpy().squeeze()
            img = np.dstack([img, img, img])
            # img = tools.colorize_cv(img)[:, :, :3]
            bb = bounding_box(skel, margin=15.0)
            bb = bb.reshape([2, 2])
            bb = np.array(bb.dot(np.array([[224 / img.shape[1], 0], [0, 224 / img.shape[0]]])), dtype=np.int32)
            bb = bb.reshape([4])
            img = cv2.resize(img, (224, 224))

            # cv2.imshow(win_name, tools.display_util.render_bb(img, bb, img_idx))
            # cv2.waitKey()

            annot_fname = os.path.join(annot_path, "{:09d}.xml".format(img_idx))
            img_fname = os.path.join(img_path, "{:09d}.png".format(img_idx))

            annot_writer = PascalVocWriter(foldername="{}_img".format(ds_type),
                                           filename="{:09d}.png".format(img_idx),
                                           imgSize=img.shape,
                                           databaseSrc='Unknown',
                                           localImgPath=img_fname)
            annot_writer.addBndBox(bb[0], bb[1], bb[2], bb[3], 'hand')
            annot_writer.save(annot_fname)
            cv2.imwrite(img_fname, img)
            img_idx += 1


if __name__ == "__main__":
    ds_output_path = "E:\\MasterDaten\\Datasets\\yolo_train"

    if os.path.exists(ds_output_path):
        shutil.rmtree(ds_output_path)

    if not os.path.exists(ds_output_path):
        os.makedirs(ds_output_path)

    with open("../datasets.json", "r") as f:
        ds_settings = json.load(f)

    ds_provider = SerializedDataset(ds_settings["BigHands"])
    prepare(ds_provider, ds_output_path)
