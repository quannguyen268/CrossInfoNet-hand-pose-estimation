#-*- coding:utf-8 -*-
#!usr/bin/python

import numpy as np
import tensorflow as tf
tf = tf.compat.v1
tf.disable_v2_behavior()
import cv2
import pyrealsense2 as rs
from Data.importers import DepthImporter
from netlib.basemodel import basenet
from util import handsegment, hand_crop
from util.realtimehandposepipeline import RealtimeHandposePipeline

import tf_slim as slim
import tf_slim.layers as layers
from imutils.video import VideoStream
import imutils

class realsense_im(object):
    def __init__(self,image_size=(640,480)):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, image_size[0], image_size[1], rs.format.z16, 30)
        config.enable_stream(rs.stream.color, image_size[0], image_size[1], rs.format.bgr8, 30)
        self.profile = self.pipeline.start(config)

    def __get_depth_scale(self):
        depth_sensor = self.profile.get_device().first_depth_sensor()

        depth_scale = depth_sensor.get_depth_scale()

        return depth_scale

    def get_image(self):

        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_image = np.asarray(depth_frame.get_data(), dtype=np.float32)
        color_image = np.asarray(color_frame.get_data(), dtype=np.uint8)
        color_image_pad = np.pad(color_image, ((20, 0), (0, 0), (0, 0)), "edge")
        depth_map_end = depth_image * self.__get_depth_scale() * 1000
        return depth_map_end,color_image_pad

    def process_end(self):
        self.pipeline.stop()


class model_setup():
    def __init__(self, dataset, model_path):
        self._dataset = dataset
        self.model_path = model_path
        self.inputs = tf.placeholder(dtype=tf.float32, shape=(None, 96, 96, 1))
        self.hand_tensor = None
        self.model()
        self.saver = tf.train.Saver()

    def __self_dict(self):
        return (21, 6, 15)

    def __config(self):
        flag=-1
        di = DepthImporter(fx=475.268, fy=flag*475.268, ux=313.821, uy=246.075)
        config = {'fx': di.fx, 'fy': abs(di.fy), 'cube': (175, 175, 175), 'im_size': (96, 96)}

        return di, config

    def __crop_cube(self):
        return self.__config()[1]['cube'][0]
    def __joint_num(self):
        return self.__self_dict()[0]

    def model(self):
        outdims = self.__self_dict()
        fn = slim.l2_regularizer(1e-5)
        fn0 = tf.no_regularizer
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=fn,
                            biases_regularizer=fn0, normalizer_fn=slim.batch_norm):
            with slim.arg_scope([slim.batch_norm],
                                is_training=False,
                                updates_collections=None,
                                decay=0.9,
                                center=True,
                                scale=True,
                                epsilon=1e-5):
                pred_comb_ht, pred_comb_hand, pred_hand, pred_ht = basenet(self.inputs, kp=1, is_training=False,
                                                                           outdims=outdims)
        self.hand_tensor = pred_hand

    def sess_run(self):
        _di, _config = self.__config()

        rtp = RealtimeHandposePipeline(1, config=_config, di=_di, verbose=False, comrefNet=None)

        joint_num = self.__joint_num()
        cube_size = self.__crop_cube()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            self.saver.restore(sess, tf.train.latest_checkpoint(self.model_path))

            depth_frame = lsa64_preprocess_image(
                '/home/quan/PycharmProjects/hand_estimation/image_test/001_010_003_frame_2.png')

            depth_frame = cv2.resize(depth_frame, (640, 480))

            frame2 = depth_frame
            crop1, M, com3D = rtp.detect(frame2)
            crop = crop1.reshape(1, crop1.shape[0], crop1.shape[1], 1).astype('float32')
            pred_ = sess.run(self.hand_tensor, feed_dict={self.inputs: crop})

            norm_hand = np.reshape(pred_, (joint_num, 3))
            pose = norm_hand * cube_size / 2. + com3D
            print(depth_frame.shape)
            img = rtp.show2(depth_frame, pose, self._dataset)
            img = rtp.addStatusBar(img)
            img = cv2.resize(img, (640, 480))
            cv2.imshow('crop', img)
            cv2.waitKey(0)



            cv2.destroyWindow()

def lsa64_preprocess_image(path):
    image = cv2.imread(path)
    image = handsegment.handsegment(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image,_ = hand_crop.one_hand(image)
    return image


if __name__=='__main__':

    model = model_setup('msra', '/home/quan/PycharmProjects/hand_estimation/model')
    model.sess_run()
    # img = lsa64_preprocess_image('/home/quan/PycharmProjects/hand_estimation/image_test/001_010_003_frame_2.png')
    # # img = cv2.resize(img, (640, 480))
    #
    # cv2.imshow('hand', img)
    # cv2.waitKey(0)