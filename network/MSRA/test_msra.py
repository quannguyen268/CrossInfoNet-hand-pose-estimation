from cffi.backend_ctypes import xrange

from Data.importers import MSRA15Importer
from util.preprocess import augmentCrop,norm_dm,joints_heatmap_gen
from util.handdetector import HandDetector
import numpy as np
from netlib.basemodel import basenet
from Data.transformations import transformPoints2D
import argparse
import matplotlib.pyplot as plt


rng=np.random.RandomState(23455)
import tensorflow as tf
tf=tf.compat.v1
tf.disable_v2_behavior()


train_root='/home/dumyy/data/msra/'
shuffle=False
di = MSRA15Importer(train_root, cacheDir='/home/quan/PycharmProjects/hand_estimation/cache/MSRA/', refineNet=None)

Seq_all=[]

Seq_test_raw=di.loadSequence('P{}'.format(0), rng=rng, shuffle=False, docom=True,cube=(175,175,175))
Seq_test=Seq_test_raw.data[:100]

test_num=len(Seq_test)
cubes_test = np.asarray([d.cube for d in Seq_test], 'float32')
coms_test = np.asarray([d.com for d in Seq_test], 'float32')
Ms_test = np.asarray([d.T for d in Seq_test], 'float32')
gt3Dcrops_test = np.asarray([d.gt3Dcrop for d in Seq_test], dtype='float32')
imgs_test = np.asarray([d.dpt.copy() for d in Seq_test], 'float32')
test_data=np.ones_like(imgs_test)
test_label=np.ones_like(gt3Dcrops_test)



for it in range(test_num):
    test_data[it]=norm_dm(imgs_test[it],coms_test[it],cubes_test[it])
    test_label[it]=gt3Dcrops_test[it]/(cubes_test[it][0]/2.)

test_data=np.expand_dims(test_data,3)
test_label=np.reshape(test_label,(-1,21*3))

hd_edges=[[0,1],[1,2],[2,3],[3,4],
          [0,5],[5,6],[6,7],[7,8],
          [0,9],[9,10],[10,11],[11,12],
          [0,13],[13,14],[14,15],[15,16],
          [0,17],[17,18],[18,19],[19,20]]
visual=True

inputs=tf.placeholder(dtype=tf.float32,shape=(None,96,96,1))
label=tf.placeholder(dtype=tf.float32,shape=(None,21*3))
gt_ht=tf.placeholder(dtype=tf.float32,shape=(None,24,24,21))
is_train=tf.placeholder(dtype=tf.bool,shape=None)
batch_size=128
last_e=100

outdims=(21,6,15)
#################################################################
import tf_slim as slim


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
        pred_comb_ht, pred_comb_hand, pred_hand, pred_ht=basenet(inputs,kp=1,is_training=is_train,outdims=outdims)



#################################################################
def getMeanError(gt, joints):
    return np.nanmean(np.nanmean(np.sqrt(np.square(gt - joints).sum(axis=2)), axis=1))

def getJointMeanError(jointID, gt, joints):
    return np.nanmean(np.sqrt(np.square(gt[:, jointID, :] - joints[:, jointID, :]).sum(axis=1)))
#################################################################

lr = tf.Variable(1e-3, dtype=tf.float32, trainable=False)
global_step = tf.Variable(0, trainable=False)

import time



gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7, allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.import_meta_graph('/home/quan/PycharmProjects/hand_estimation/model/crossInfoNet_MSRA0.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('/home/quan/PycharmProjects/hand_estimation/model/'))

    pred_norm=[]
    loopv = test_num // batch_size
    other = test_data[loopv * batch_size:]
    a=time.time()
    for itest in xrange(loopv + 1):
        if itest < loopv:
            start = itest * batch_size
            end = (itest + 1) * batch_size
            feed_dict = {inputs: test_data[start:end],is_train:False}
        else:
            feed_dict = {inputs: other,is_train:False}
        [pred_] = sess.run([pred_hand], feed_dict=feed_dict)
        pred_norm.append(pred_)
    b=time.time()
    print ('frame test time: ',b-a)
    norm_hands = np.concatenate(pred_norm, 0).reshape(-1, 21, 3)
    pred_hands = norm_hands * np.tile(np.expand_dims(cubes_test / 2., 1), (1, 21, 1)) + \
                 np.tile(np.expand_dims(coms_test, 1),(1, 21, 1))
    gt_hands = test_label.reshape(-1, 21, 3) * np.tile(np.expand_dims(cubes_test / 2., 1), (1, 21, 1)) + \
                 np.tile(np.expand_dims(coms_test, 1), (1, 21, 1))
    meane = getMeanError(gt_hands, pred_hands)
print ('mean error: ', meane)
print (getJointMeanError(j,gt_hands,pred_hands) for j in range(21))
#
#
# f = open('res_msra_{}.txt'.format(0), 'a+')
# for i in range(pred_hands.shape[0]):
#     uvds=di.joints3DToImg(pred_hands[i])
#     uvds=np.reshape(uvds,(1,63))
#     for j in range(63):
#         f.write(str(round(uvds[0,j],4)))
#         f.write(' ')
#     f.write('\n')
# f.close()


if visual==True:
    import matplotlib.pyplot as plt
    for i in range(0,test_num,10):
        plt.imshow(np.squeeze(test_data[i]), cmap='gray')
        jtIp = transformPoints2D(di.joints3DToImg(pred_hands[i]), Ms_test[i])
        plt.scatter(jtIp[:, 0], jtIp[:, 1], c='r')

        jtIt = transformPoints2D(di.joints3DToImg(gt_hands[i]), Ms_test[i])
        plt.scatter(jtIt[:, 0], jtIt[:, 1], c='b')

        for edge in hd_edges:
            plt.plot(jtIp[:, 0][edge], jtIp[:, 1][edge], c='r',linewidth=1.5)
            plt.plot(jtIt[:, 0][edge], jtIt[:, 1][edge], c='b',linewidth=1.5)
        plt.pause(0.01)

        plt.axis('off')

        plt.cla()




