import pickle
from Data.importers import MSRA15Importer
from util.preprocess import augmentCrop, norm_dm, joints_heatmap_gen
from util.handdetector import HandDetector
import numpy as np
from netlib.basemodel import basenet
from Data.transformations import transformPoints2D
import tensorflow as tf

tf = tf.compat.v1
tf.disable_v2_behavior()

train_root = '/home/quan/Downloads/cvpr14_MSRAHandTrackingDB'
shuffle = True
rng = np.random.RandomState(23455)
msra = MSRA15Importer(train_root, cacheDir='/home/quan/PycharmProjects/hand_estimation/cache/MSRA', refineNet=None)
# %%
Seq_all = []
for seq in [0, 1,4,5]:
    shuffle = True
    if seq == 0:
        Seq_train_ = msra.loadSequence('P{}'.format(seq), rng=rng, shuffle=shuffle, docom=True, cube=(175, 175, 175))
    else:
        Seq_train_ = msra.loadSequence('P{}'.format(seq), rng=rng, shuffle=shuffle, docom=True, cube=None)
    Seq_all.append(Seq_train_)

Seq_test_raw = Seq_all.pop(0)
Seq_test = Seq_test_raw.data
Seq_train = [seq_data for seq_ in Seq_all for seq_data in seq_.data]
rng.shuffle(Seq_train)
Seq_test = Seq_test[:500]




# %%
train_num = len(Seq_train)
cubes_train = np.asarray([d.cube for d in Seq_train], 'float32')
coms_train = np.asarray([d.com for d in Seq_train], 'float32')
Ms_train = np.asarray([d.T for d in Seq_train], dtype='float32')
gt3Dcrops_train = np.asarray([d.gt3Dcrop for d in Seq_train], dtype='float32')
imgs_train = np.asarray([d.dpt for d in Seq_train], 'float32')

test_num = len(Seq_test)
cubes_test = np.asarray([d.cube for d in Seq_test], 'float32')
coms_test = np.asarray([d.com for d in Seq_test], 'float32')
gt3Dcrops_test = np.asarray([d.gt3Dcrop for d in Seq_test], dtype='float32')
imgs_test = np.asarray([d.dpt for d in Seq_test], 'float32')
Ms_test = np.asarray([d.T for d in Seq_test], 'float32')

test_data = np.ones_like(imgs_test)
test_label = np.ones_like(gt3Dcrops_test)

print("training data {}".format(imgs_train.shape[0]))
print("testing data {}".format(imgs_test.shape[0]))
print("testing sub {}".format(0))
print("done")

for it in range(test_num):
    test_data[it] = norm_dm(imgs_test[it], coms_test[it], cubes_test[it])
    test_label[it] = gt3Dcrops_test[it] / (cubes_test[it][0] / 2.)
test_data = np.expand_dims(test_data, 3)
test_label = np.reshape(test_label, (-1, 21 * 3))

# %%
hd_edges = [[0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8],
            [0, 9], [9, 10], [10, 11], [11, 12],
            [0, 13], [13, 14], [14, 15], [15, 16],
            [0, 17], [17, 18], [18, 19], [19, 20]]
visual = False
visual_aug = False
if visual == True:
    import matplotlib.pyplot as plt

    for i in range(0, test_num, 10):
        plt.imshow(imgs_test[i], cmap='gray')
        gt3D = gt3Dcrops_test[i] + coms_test[i]
        jtI = transformPoints2D(msra.joints3DToImg(gt3D), Ms_test[i])
        plt.scatter(jtI[:, 0], jtI[:, 1])
        for edge in hd_edges:
            plt.plot(jtI[:, 0][edge], jtI[:, 1][edge], c='r')
        plt.pause(0.001)
        plt.cla()

# %%
hd = HandDetector(imgs_train[0].copy(), abs(msra.fx), abs(msra.fy), importer=msra, refineNet=None)
inputs = tf.placeholder(dtype=tf.float32, shape=(None, 96, 96, 1))
label = tf.placeholder(dtype=tf.float32, shape=(None, 21 * 3))
gt_ht = tf.placeholder(dtype=tf.float32, shape=(None, 24, 24, 21))
is_train = tf.placeholder(dtype=tf.bool, shape=None)
kp = tf.placeholder(dtype=tf.float32, shape=None)

batch_size = 32
last_e = 100
outdims = (21, 6, 15)

# %%
import tf_slim as slim
from netutil.util import process

LOG_DIR = '/home/quan/PycharmProjects/hand_estimation/logs/CrossInfoNet_with_summaries'


# Add some ops that take care of logging summaries
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


fn = slim.l2_regularizer(1e-5)
fn0 = tf.no_regularizer
global_step = tf.Variable(0, trainable=False)

with tf.name_scope('learning_rate'):
    lr = tf.Variable((1e-3), dtype=tf.float32, trainable=False)
    variable_summaries(lr)
    tf.summary.scalar("learing_rate", lr)



with tf.name_scope('Model'):
    with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=fn,
                        biases_regularizer=fn0, normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm],
                            is_training=is_train,
                            updates_collections=None,
                            decay=0.9,
                            center=True,
                            scale=True,
                            epsilon=1e-5):
            pred_comb_ht, pred_comb_hand, pred_hand, pred_ht = basenet(inputs, kp=kp, is_training=is_train,
                                                                       outdims=outdims)

# %%
with tf.name_scope('Euclidean_loss') as scope:
    with tf.name_scope('ground_truth'):
        gt_palm_ht = tf.concat((gt_ht[:, :, :, 0:1], gt_ht[:, :, :, 1::4]), 3)
        gt_fing_ht = tf.concat((gt_ht[:, :, :, 2::4], gt_ht[:, :, :, 3::4], gt_ht[:, :, :, 4::4]), 3)

        label1 = tf.reshape(label, (-1, 21, 3), name='reshape_1')
        gt_fing = tf.reshape(tf.concat((label1[:, 2::4, :], label1[:, 3::4, :], label1[:, 4::4, :]), 1), (-1, 15 * 3),
                             name='reshape1_gt_fing')
        gt_palm = tf.reshape(tf.concat((label1[:, 0:1, :], label1[:, 1::4, :]), 1), (-1, 6 * 3),
                             name='reshape1_gt_palm')

    with tf.name_scope('loss_heatmap'):
        loss_ht = tf.nn.l2_loss((pred_ht - gt_ht), name='loss_ht') / batch_size
        loss_palm_ht = tf.nn.l2_loss((pred_comb_ht[0] - gt_palm_ht), name='loss_palm_ht') / batch_size
        loss_fing_ht = tf.nn.l2_loss((pred_comb_ht[1] - gt_fing_ht), name='loss_fing_ht') / batch_size

    with tf.name_scope('loss_regression'):
        loss_hand = tf.nn.l2_loss((pred_hand - label), name='loss_hand') / batch_size
        loss_palm = tf.nn.l2_loss((pred_comb_hand[0] - gt_palm), name='loss_palm') / batch_size
        loss_fing = tf.nn.l2_loss((pred_comb_hand[1] - gt_fing), name='loss_fing') / batch_size

    with tf.name_scope('weight_decay'):
        weight_decay = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        weight_decay = tf.add_n(weight_decay, name='weight_add')
        tf.summary.scalar("weight_decay", weight_decay)

    with tf.name_scope('total_loss'):
        total_loss = 0.5 * ((loss_ht + loss_palm_ht + loss_fing_ht) * 0.1 + loss_hand + loss_palm + loss_fing) + weight_decay
        tf.summary.scalar("loss", total_loss)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)


def getMeanError(gt, joints):
    return np.nanmean(np.nanmean(np.sqrt(np.square(gt - joints).sum(axis=2)), axis=1))


summ = tf.summary.merge_all()
saver = tf.train.Saver()
# %%

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    saver.restore(sess, tf.train.latest_checkpoint('/home/quan/PycharmProjects/hand_estimation/model/'))

    # Write summaries to LOG_DIR -- used by Tensorboard
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', graph=sess.graph)
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test', graph=sess.graph)

    kpv = 0.6
    j = 0
    step = 0
    iteration = 100

    for itrain in range((train_num // batch_size) * iteration):

        if itrain % ((train_num// batch_size) * 2) == 0:
            lr_update = tf.assign(lr, (1e-3) * 0.96 ** step)
            sess.run(lr_update)
            step += 1

        subdata = imgs_train[j: j + batch_size]
        subcom = coms_train[j: j + batch_size]
        subcube = cubes_train[j: j + batch_size]
        subM = Ms_train[j: j + batch_size]
        subgt3Dcrop = gt3Dcrops_train[j: j + batch_size]
        resdata = np.ones_like(subdata)
        resgt3D = np.ones_like(subgt3Dcrop)
        hts = np.zeros(shape=(batch_size, 24, 24, 21))

        j = (j + batch_size) % train_num
        if (j + batch_size) >= train_num:
            j = j + batch_size - train_num
        for idx in range(batch_size):
            dm = norm_dm(subdata[idx], subcom[idx], subcube[idx])
            s = augmentCrop(dm, subgt3Dcrop[idx], msra.joint3DToImg(subcom[idx]),
                            subcube[idx], subM[idx], ['rot', 'sc', 'com', 'none'], hd, False, rng=rng)
            resdata[idx] = s[0]
            resgt3D[idx] = s[2]
            mode = s[7]
            gt3D_ = resgt3D[idx] * subcube[idx][0] / 2. + subcom[idx]
            jtI_ = transformPoints2D(msra.joints3DToImg(gt3D_), subM[idx])

            jtI_ = np.reshape(jtI_, (1, 21 * 3))
            ht_ = joints_heatmap_gen([1], jtI_, (24, 24), points=21)
            hts[idx] = np.transpose(ht_, (0, 2, 3, 1)) / 255.


        resdata = np.reshape(resdata, (-1, 96, 96, 1))
        resgt3D = np.reshape(resgt3D, (-1, 21 * 3))
        _, summs = sess.run([optimizer, summ],
                                   feed_dict={inputs: resdata,
                                              label: resgt3D,
                                              gt_ht: hts,
                                              is_train: True,
                                              kp: kpv})
        train_writer.add_summary(summs, step)



        if itrain % (train_num // (batch_size*5)) == 0:

            pred_norm = []
            loopv = test_num // batch_size
            other = test_data[loopv * batch_size:]
            for itest in range(loopv + 1):
                if itest < loopv:
                    start = itest * batch_size
                    end = (itest + 1) * batch_size
                    feed_dict = {inputs: test_data[start:end], is_train: False, kp: kpv}
                else:
                    feed_dict = {inputs: other, is_train: False, kp: kpv}
                [pred_] = sess.run([pred_hand], feed_dict=feed_dict)
                pred_norm.append(pred_)
            norm_hands = np.concatenate(pred_norm, 0).reshape(-1, 21, 3)
            pred_hands = norm_hands * np.tile(np.expand_dims(cubes_test / 2., 1), (1, 21, 1)) + \
                         np.tile(np.expand_dims(coms_test, 1), (1, 21, 1))
            gt_hands = test_label.reshape(-1, 21, 3) * np.tile(np.expand_dims(cubes_test / 2., 1), (1, 21, 1)) + \
                       np.tile(np.expand_dims(coms_test, 1), (1, 21, 1))
            meane = getMeanError(gt_hands, pred_hands)

            print("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(meane) )
            logt = open('/home/quan/PycharmProjects/hand_estimation/logs/CrossInfoNet_with_summaries/mean.error.epoch/logt_msra_{}.txt'.format(0), 'a+')
            logt.write('epoch {}, mean error {}'.format(step, meane))
            logt.write('\n')
            logt.close()


            saver.save(sess, '/home/quan/PycharmProjects/hand_estimation/model/crossInfoNet_MSRA{}.ckpt'.format(0))
