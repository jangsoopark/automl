from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib

from lxml import etree

import tensorflow.compat.v1 as tf
import json
import glob
import PIL
import csv
import cv2
import os
import io

import dataset.jangsoopark.pascal.pascal_voc_writer as pascal_writer
import dataset.tfrecord_util as tfrecord_util

GLOBAL_IMG_ID = 0  # global image id
GLOBAL_ANN_ID = 0  # global annotation id


def parse_yolo_labels(image_list_path: str, classes: list):
    f = open(image_list_path)
    for line in f:
        image_file_name = line.strip()

        image = cv2.imread(image_file_name)
        h, w, _ = image.shape

        writer = pascal_writer.PascalVocWriter(image_file_name, width=w, height=h)

        current_labels = []

        label_file_name = image_file_name.replace('jpg', 'txt')
        xml_file_name = image_file_name.replace('jpg', 'xml')

        with open(label_file_name) as csv_file:
            labels = csv.reader(csv_file, delimiter=' ')
            for row in labels:
                class_id = int(row[0]) + 1
                rw = int(float(row[3]) * w)
                rh = int(float(row[4]) * h)
                cx = int(float(row[1]) * w)
                cy = int(float(row[2]) * h)

                xmin = cx - (rw >> 1)
                ymin = cy - (rh >> 1)
                xmax = xmin + rw
                ymax = ymin + rh

                image_name = os.path.split(image_file_name)

                box = [class_id, xmin, ymin, xmax, ymax]
                current_labels.append(box)

                writer.add_object(
                    classes[class_id], xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, difficult=1
                )

            writer.save(xml_file_name)

    f.close()


def get_image_id():
    global GLOBAL_IMG_ID
    GLOBAL_IMG_ID += 1
    return GLOBAL_IMG_ID


def get_annotation_id():
    global GLOBAL_ANN_ID
    GLOBAL_ANN_ID += 1
    return GLOBAL_ANN_ID


def dict_to_tf_example(data, dataset_directory, label_map, ignore_difficult_instance, ann_json):
    image_path = os.path.join(data['folder'], data['filename'])
    full_path = os.path.join(dataset_directory, image_path)
    with tf.gfile.GFile(full_path, 'rb') as f:
        encoded_image = f.read()
    #
    # encoded_image_io = io.BytesIO(encoded_image)
    # image = PIL.Image.open(encoded_image_io)

    key = hashlib.sha256(encoded_image).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])
    image_id = get_image_id()

    image = {
        'file_name': data['filename'],
        'height': height,
        'width': width,
        'id': image_id
    }
    ann_json['images'].append(image)

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []

    if 'object' in data:
        for obj in data['object']:

            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instance and difficult:
                continue

            difficult_obj.append(int(difficult))
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)

            classes_text.append(obj['name'].encode('utf-8'))
            classes.append(label_map[obj['name']])

            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf-8'))

            if ann_json:
                abs_xmin = int(obj['bndbox']['xmin'])
                abs_ymin = int(obj['bndbox']['ymin'])
                abs_xmax = int(obj['bndbox']['xmax'])
                abs_ymax = int(obj['bndbox']['ymax'])

                abs_width = abs_xmax - abs_xmin
                abs_height = abs_ymax - abs_ymin

                ann = {
                    'area': abs_width * abs_height,
                    'iscrowd': 0,
                    'image_id': image_id,
                    'bbox': [abs_xmin, abs_ymin, abs_width, abs_height],
                    'category_id': label_map[obj['name']],
                    'id': get_annotation_id(),
                    'ignore': 0,
                    'segmentation': [],
                }
                ann_json['annotations'].append(ann)

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height': tfrecord_util.int64_feature(height),
                'image/width': tfrecord_util.int64_feature(width),
                'image/filename': tfrecord_util.bytes_feature(data['filename'].encode('utf8')),
                'image/source_id': tfrecord_util.bytes_feature(str(image_id).encode('utf8')),
                'image/key/sha256': tfrecord_util.bytes_feature(key.encode('utf8')),
                'image/encoded': tfrecord_util.bytes_feature(encoded_image),
                'image/format': tfrecord_util.bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin': tfrecord_util.float_list_feature(xmin),
                'image/object/bbox/xmax': tfrecord_util.float_list_feature(xmax),
                'image/object/bbox/ymin': tfrecord_util.float_list_feature(ymin),
                'image/object/bbox/ymax': tfrecord_util.float_list_feature(ymax),
                'image/object/class/text': tfrecord_util.bytes_list_feature(classes_text),
                'image/object/class/label': tfrecord_util.int64_list_feature(classes),
                'image/object/difficult': tfrecord_util.int64_list_feature(difficult_obj),
                'image/object/truncated': tfrecord_util.int64_list_feature(truncated),
                'image/object/view': tfrecord_util.bytes_list_feature(poses),
            }))
    return example


if __name__ == '__main__':
    num_shards = 50
    mode = 'test'  # test
    tfrecord_prefix = 'D:\\ivs\\AILAB\\efficientdet\\tfrecords\\style3k-' + mode
    tfrecord_patterns = '-%05d-of-%05d.tfrecord'

    annotations_pattern = os.path.join(
        os.path.dirname(os.path.dirname(tfrecord_prefix)), 'images', '*.xml')
    data_root = os.path.dirname(os.path.dirname(tfrecord_prefix))

    _classes = ['background', 'athleisure', 'chiccasual', 'pret-a-couture']

    parse_yolo_labels('%s.txt' % mode, _classes)

    if not tf.io.gfile.exists(os.path.dirname(tfrecord_prefix)):
        tf.io.gfile.mkdir(os.path.dirname(tfrecord_prefix))

    writers = [
        tf.python_io.TFRecordWriter(
            tfrecord_prefix + tfrecord_patterns % (i, num_shards)
        ) for i in range(num_shards)
    ]

    ann_json_dict = {
        'images': [],
        'type': 'instances',
        'annotations': [],
        'categories': []
    }

    for class_name in _classes:
        cls = {'supercategory': 'none', 'id': _classes.index(class_name), 'name': class_name}
        ann_json_dict['categories'].append(cls)

    annotations_list = glob.glob(annotations_pattern)

    for idx, path in enumerate(annotations_list):
        with tf.gfile.GFile(path, 'r') as f:
            xml_str = f.read()
        print(path)
        xml = etree.fromstring(xml_str)
        data = tfrecord_util.recursive_parse_xml_to_dict(xml)['annotation']
        tf_example = dict_to_tf_example(
            data,
            data_root,
            dict(zip(_classes, [_classes.index(c) for c in _classes])),
            False,
            ann_json=ann_json_dict
        )

        writers[idx % num_shards].write(tf_example.SerializeToString())

    for writer in writers:
        writer.close()

    json_file_path = os.path.join(
        os.path.dirname(tfrecord_prefix),
        os.path.basename(tfrecord_prefix) + '.json'
    )

    with tf.io.gfile.GFile(json_file_path, 'w') as f:
        json.dump(ann_json_dict, f)
