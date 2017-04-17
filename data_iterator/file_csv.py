import os, os.path
import shutil, string
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

'''
### record list as CSV
fileinfo = open('list.csv', 'w')
for i in fileList:
    curname = os.path.join(dir, i)
    print curname
    fileinfo.write(curname + label + '\n')
fileinfo.close()
'''

Map_size = 28


def convert_and_save(filename):
    dir = '../loam'
    fileList = os.listdir(dir)
    fileList.sort()
    writer = tf.python_io.TFRecordWriter(filename)
    for i in fileList:
        curname = os.path.join(dir, i)
        img = Image.open(curname).convert('L')
        img = img.resize((Map_size, Map_size))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))
        writer.write(example.SerializeToString())

    writer.close()

def read_and_decode(filename_queue):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={'img_raw': tf.FixedLenFeature([], tf.string),})

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [Map_size, Map_size, 1])
    return img

def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        [filenames],shuffle=True)
    example = read_and_decode(filename_queue)
    min_after_dequeue = 400
    capacity = min_after_dequeue + 3 * batch_size
    example_batch  = tf.train.shuffle_batch(
        [example], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch

def output_image(filename):
    example_batch = input_pipeline(filename, 32, 1)

    # Limit GPU usage
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth=True
    np.set_printoptions(threshold='nan')


    with tf.Session(config=config) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(1000):
            example = sess.run(example_batch)
            print example.shape
        coord.request_stop()
        coord.join(threads)
    return example

if __name__ == '__main__':
    convert_and_save('test.tfrecords')
    #output_image('test.tfrecords')
