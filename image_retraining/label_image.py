# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

Run image classification with your model.

This script is usually used with retrain.py found in this same
directory.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. You are required
to pass in the graph file and the txt file.

It outputs human readable strings of the top 5 predictions along with
their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Example usage:
python label_image.py --graph=retrained_graph.pb
  --labels=retrained_labels.txt
  --image=flower_photos/daisy/54377391_15648e8d18.jpg

NOTE: To learn to use this file and retrain.py, please see:

https://codelabs.developers.google.com/codelabs/tensorflow-for-poets
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib

import argparse
import os
import sys

import tensorflow as tf


def load_image(filename):
    """Read in the image_data to be classified."""
    return tf.gfile.FastGFile(filename, 'rb').read()


def load_labels(filename):
    """Read in labels, one label per line."""
    return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
    """Unpersists graph from file as default graph."""
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def run_graph(image_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
    result = {}
    with tf.Session() as sess:
        # Feed the image_data as input to the graph.
        #   predictions will contain a two-dimensional array, where one
        #   dimension represents the input image count, and the other has
        #   predictions per class
        softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
        predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})

        # Sort to show labels in order of confidence
        top_k = predictions.argsort()[-num_top_predictions:][::-1]

        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%s[%.5f]' % (name_mapping[human_string], score), end='<br>')
            result[human_string] = score

        return result


def download_image(url):
    # 需要安装wget  和 convert
    out_file = '/tmp/%s.jpg' % hashlib.md5(bytes(str(url).encode('utf-8'))).hexdigest()
    if os.path.exists(out_file):
        tf.logging.info('image file exist skip download %s', url)
        return out_file

    print(os.popen('/usr/local/bin/wget  -O /tmp/image.bin %s;'
                   '/opt/local/bin/convert /tmp/image.bin %s' % (
                       url, out_file)).read())
    return out_file
    pass


name_mapping = {'shuaige': '帅哥',
                'chounan': '丑男',
                'meinv': '美女',
                'chounv': '丑女',
                'xinggan': '性感美女',
                'dachangtui': '长腿美女'
                }


def main(argv):
    """Runs inference on an image."""

    print(FLAGS)
    # load image
    if argv[1:]:
        raise ValueError('Unused Command Line Args: %s' % argv[1:])
    image_path = FLAGS.image
    if str(image_path).startswith("http"):
        image_path = download_image(image_path)

    print(image_path)
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('image file does not exist %s', image_path)

    if not tf.gfile.Exists(FLAGS.labels):
        tf.logging.fatal('labels file does not exist %s', FLAGS.labels)

    if not tf.gfile.Exists(FLAGS.graph):
        tf.logging.fatal('graph file does not exist %s', FLAGS.graph)

    image_data = load_image(image_path)

    # load labels
    labels = load_labels(FLAGS.labels)

    # load graph, which is stored in the default session
    load_graph(FLAGS.graph)

    result = run_graph(image_data, labels, FLAGS.input_layer, FLAGS.output_layer,
                       FLAGS.num_top_predictions)

    shuaige_score = result.get('shuaige', 0)
    chounan_score = result.get('chounan', 0)
    meinv_score = result.get('meinv', 0)
    chounv_score = result.get('chounv', 0)
    xinggan_score = result.get('xinggan', 0)

    print("============分析结果=============")
    if shuaige_score + chounan_score > meinv_score + chounv_score and xinggan_score < (
            shuaige_score if shuaige_score > chounan_score else chounan_score):
        print("%s男生 颜值=%.2f" % (('应该是' if shuaige_score > 0.3  else '可能是'),
                                100 * shuaige_score / (shuaige_score + chounan_score)))

    else:
        extra = ('应该是' if meinv_score or xinggan_score > 0.3  else '可能是')

        if xinggan_score > 0.1:
            extra += '性感'
        print("%s女生  \t颜值=%.2f" % (extra, 100 * meinv_score / (meinv_score + chounv_score)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image',
        type=str,
        # default='/Users/haizhu/Desktop/jiemo/baco.jpg',
        default='http://pic2016.ytqmx.com:82/2016/0516/19/1.jpg!960.jpg',
        help='Absolute path to image file.')
    parser.add_argument(
        '--graph',
        default='data/output_graph.pb',
        type=str,
        help='Absolute path to graph file (.pb)')
    parser.add_argument(
        '--labels',
        type=str,
        default='data/output_labels.txt',
        help='Absolute path to labels file (.txt)')
    parser.add_argument(
        '--output_layer',
        type=str,
        default='final_result:0',
        help='Name of the result operation')
    parser.add_argument(
        '--input_layer',
        type=str,
        default='DecodeJpeg/contents:0',
        help='Name of the input operation')
    parser.add_argument(
        '--num_top_predictions',
        type=int,
        default=5,
        help='Display this many predictions.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=sys.argv[:1] + unparsed)
