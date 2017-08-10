# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile
import time
from urllib import request
from urllib.error import URLError

import numpy as np
import tensorflow as tf
from six.moves import urllib

FLAGS = None

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


# pylint: enable=line-too-long


class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
        cn_uid_lookup_path = os.path.join(
            '../../../data', 'imagenet_synset_to_human_label_map_CN.txt')
        self.cn_node_id_to_name = {}
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path, cn_uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path, cn_uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.

        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # load replace by china name
        # 使用已经有的中文翻译覆盖掉英文
        proto_as_ascii_lines = tf.gfile.GFile(cn_uid_lookup_path).readlines()
        p = re.compile(r'[n\d]*[ \S,]*')
        cn_uid_to_human = {}
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            cn_uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        cn_node_id_to_name = {}

        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name
            cn_node_id_to_name[key] = cn_uid_to_human[val]
        self.cn_node_id_to_name = cn_node_id_to_name
        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return str(self.cn_node_id_to_name[node_id]) + '[' + self.node_lookup[node_id] + ']'


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def init_image_classify_names():
    """Runs inference on an image.

    Args:
      image: Image file name.

    Returns:
      Nothing
    """
    # Creates graph from saved GraphDef.
    create_graph()

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    return node_lookup


def classify_by_image(image, node_lookup):
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    start = time.time()
    results = {}

    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
        # result = ''
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            # item = '%s (score = %.5f)' % (human_string, score)
            # print(item)
            # result = result + ' ' + item
            results[human_string] = score
            # print(result)

    print(time.time() - start)
    return results


def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def download(url):
    headers = {'User-agent':
                   'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'
               # 'Mozilla/5.0 (iPhone; CPU iPhone OS 9_1 like Mac OS X) AppleWebKit/601.1.46 (KHTML, like Gecko) Version/9.0 Mobile/13B143 Safari/601.1'
               }
    file = None
    try:
        req = request.Request(url, headers=headers)
        response = request.urlopen(req, timeout=10)

        content_type = response.getheader('Content-Type')

        ext = ''
        if content_type is not None:
            ct = str(content_type)
            start = ct.find('/')
            if start > 0:
                ext = "." + ct[(start + 1):]

        print(content_type)
        page = response.read()

        file = os.path.join(FLAGS.model_dir, "temp_image" + (ext))
        with open(file, "wb") as f:
            f.write(page)

    except (URLError, ValueError, IOError, Exception) as err:
        err_info = str(err)
        print(err_info)
    finally:
        return file


def region_image(image, node_lookup):
    # 不能用webp需要转换 只适用于aliyun，其他的请自己修改  或者用本地图片
    if image.startswith("http"):
        if image.find('jiemosrc') > 0 and image.endswith("webp"):

            if image.endswith(",webp"):
                image = image[:len(image) - 4] + "jpeg"
            elif image.endswith(".webp"):
                image = image + "?x-oss-process=image/format,jpeg"

        image_path = download(image)
    elif image.startswith('/'):
        image_path = image
    else:
        image_path = (FLAGS.image_file if FLAGS.image_file else
                      os.path.join(FLAGS.model_dir, '1.png'))

    results = classify_by_image(image_path, node_lookup)
    for k, v in results.items():
        print('%s\t%s\t%s ' % (image, k, v))
    print('\n\n\n\n')
    pass


def main(_):
    maybe_download_and_extract()
    node_lookup = init_image_classify_names()
    region_image('/Users/haizhu/Downloads/tensorflow/images/cropped_panda.jpg', node_lookup)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # classify_image_graph_def.pb:
    #   Binary representation of the GraphDef protocol buffer.
    # imagenet_synset_to_human_label_map.txt:
    #   Map from synset ID to a human readable string.
    # imagenet_2012_challenge_label_map_proto.pbtxt:
    #   Text representation of a protocol buffer mapping a label to synset ID.
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/Users/haizhu/Downloads/tensorflow/images',
        help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
    )
    parser.add_argument(
        '--image_file',
        type=str,
        default='/test.jpg',
        help='Absolute path to image file.'
    )
    parser.add_argument(
        '--num_top_predictions',
        type=int,
        default=5,
        help='Display this many predictions.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
