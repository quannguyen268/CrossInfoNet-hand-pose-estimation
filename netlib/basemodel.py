import tensorflow as tf
import tf_slim as slim
from netutil import resnet_v1, resnet_utils


tf = tf.compat.v1
tf.disable_v2_behavior()
bottleneck = resnet_v1.bottleneck

def basenet(inp, kp=0.5, is_training=True, outdims=(21, 6, 15)):
    '''
    :param inp: input data
    :param kp: dropout keep rate
    :param is_training: is training?
    :param outdims: (hand_num, palm_num, finger_num)
    :return: output
    '''

    with tf.name_scope('bone_net'):
        blocks = [
            resnet_v1.resnet_v1_block('block1', base_depth=16, num_units=3, stride=2),
            resnet_v1.resnet_v1_block('block2', base_depth=32, num_units=4, stride=2),
            resnet_v1.resnet_v1_block('block3', base_depth=64, num_units=6, stride=2),
            resnet_v1.resnet_v1_block('block4', base_depth=64, num_units=3, stride=2),
        ]
        net = resnet_utils.conv2d_same(inp, 32, 5, stride=2, scope='conv1')
        # net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='pool1')

        net1, _ = resnet_v1.resnet_v1(net, blocks[0:1], scope='nn1', is_training=False, global_pool=False,
                                      include_root_block=False)
        net2, _ = resnet_v1.resnet_v1(net1, blocks[1:2], scope='nn2', is_training=False, global_pool=False,
                                      include_root_block=False)
        net3, _ = resnet_v1.resnet_v1(net2, blocks[2:3], scope='nn3', is_training=False, global_pool=False,
                                      include_root_block=False)
        net4, _ = resnet_v1.resnet_v1(net3, blocks[3:4], scope='nn4', is_training=False, global_pool=False,
                                      include_root_block=False)
        feature_maps = [net1, net2, net3, net4]

    with tf.name_scope("elem_net"):
        global_fms = []
        last_fm = None
        initializer = slim.xavier_initializer()

        with tf.name_scope("global_net"):
            # net4, net3, net2, net1
            for i, block in enumerate(reversed(feature_maps)):
                lateral = slim.conv2d(block, 256, [1, 1],
                                      weights_initializer=initializer,
                                      padding='SAME', activation_fn=tf.nn.relu,
                                      scope='lateral/res{}'.format(5 - i))
                if last_fm is not None:
                    sz = tf.shape(lateral)
                    upsample = tf.image.resize(last_fm, (sz[1], sz[2]), method=tf.image.ResizeMethod.BILINEAR,
                                                        name='upsample/res{}'.format(5 - i))
                    upsample = slim.conv2d(upsample, 256, [1, 1],
                                           weights_initializer=initializer,
                                           padding='SAME', activation_fn=None,
                                           scope='merge/res{}'.format(5 - i))
                    last_fm = upsample + lateral
                else:
                    last_fm = lateral
                global_fms.append(last_fm)
                # conv -> add -> add -> add
        global_fms.reverse()
        # add -> add -> add -> conv

    with tf.name_scope('heatmap'):
        ht_map = global_fms[-4]
        ht_map = bottleneck(ht_map, 256, 128, stride=1, scope='htmap_bottleneck')
        ht_out = slim.conv2d(ht_map, num_outputs=outdims[0], kernel_size=(3, 3), stride=1, activation_fn=None, scope='ht_out')

    with tf.name_scope('cacsed'):
        hand_map_ = global_fms[-3]

        # ==> PALM MAP <== #
        palm_map = bottleneck(hand_map_, 256, 128, stride=1, scope='palm_bottleneck')

        ht_palm = slim.conv2d(palm_map, 256, 1, 1, activation_fn=tf.nn.relu, scope='ht_palm')
        ht_palm_out_ = slim.conv2d(ht_palm, num_outputs=outdims[1], kernel_size=(3, 3), activation_fn=None, scope='ht_palm_out_')
        ht_palm_out = tf.image.resize(ht_palm_out_, (24, 24), method=tf.image.ResizeMethod.BILINEAR)
        ###

        # ==> FINGER MAP <== #
        fing_map = bottleneck(hand_map_, 256, 128, stride=1, scope='fing_bottleneck')

        ht_fing = slim.conv2d(palm_map, 256, 1, 1, activation_fn=tf.nn.relu, scope='ht_fing')
        ht_fing_out_ = slim.conv2d(ht_fing, num_outputs=outdims[2], kernel_size=(3, 3), activation_fn=None, scope='ht_fing_out_')
        ht_fing_out = tf.image.resize(ht_fing_out_, (24, 24), method=tf.image.ResizeMethod.BILINEAR)
        ###

        # ==> FINGER REGRESSION  <== #
        res_fing_map = hand_map_ - palm_map
        end_fing_map = tf.concat([fing_map, res_fing_map], axis=3, name='concat_fing_map')

        end_fing_map_ = bottleneck(end_fing_map, 256, 128, stride=1, scope='end_fing_map')
        end_fing_map_pooling = slim.max_pool2d(end_fing_map_, 2)

        end_fing_ = slim.flatten(end_fing_map_pooling, scope='reg_end_fing_0')
        end_fing_ = slim.fully_connected(end_fing_, 1024, activation_fn=tf.nn.relu, scope='reg_end_fing_1')
        end_fing_ = slim.dropout(end_fing_, keep_prob=kp, is_training=is_training, scope='reg_end_fing_2')
        end_fing_ = slim.fully_connected(end_fing_, 1024, activation_fn=tf.nn.relu, scope='reg_end_fing_3')
        end_fing_ = slim.dropout(end_fing_, keep_prob=kp, is_training=is_training, scope='reg_end_fing_4')
        end_fing_out = slim.fully_connected(end_fing_, num_outputs=outdims[2] * 3, activation_fn=None, scope='reg_end_fing_out')
        ###

        # ==> PALM REGRESSION <== #
        res_palm_map = hand_map_ - palm_map
        end_palm_map = tf.concat([palm_map, res_palm_map], axis=3, name='concat_palm_map')

        end_palm_map_ = bottleneck(end_palm_map, 256, 128, stride=1, scope='end_palm_map')
        end_palm_map_pooling = slim.max_pool2d(end_palm_map_, 2)

        end_palm_ = slim.flatten(end_palm_map_pooling, scope='reg_end_palm_0')
        end_palm_ = slim.fully_connected(end_palm_, 1024, activation_fn=tf.nn.relu, scope='reg_end_palm_1')
        end_palm_ = slim.dropout(end_palm_, keep_prob=kp, is_training=is_training, scope='reg_end_palm_2')
        end_palm_ = slim.fully_connected(end_palm_, 1024, activation_fn=tf.nn.relu, scope='reg_end_palm_3')
        end_palm_ = slim.dropout(end_palm_, keep_prob=kp, is_training=is_training, scope='reg_end_palm_4')
        end_palm_out = slim.fully_connected(end_palm_, num_outputs=outdims[1] * 3, activation_fn=None, scope='reg_end_palm_out')
        ###

        end_hand = tf.concat([end_palm_, end_fing_], axis=1, name='concat_endHand')
        end_hand_out = slim.fully_connected(end_hand, num_outputs=outdims[0] * 3, activation_fn=None, scope='end_hand_out')

        comb_ht_out = [ht_palm_out, ht_fing_out]
        comb_hand_out = [end_palm_out, end_fing_out]
        hand_out = end_hand_out

        return comb_ht_out, comb_hand_out, hand_out, ht_out


from tensorflow.keras.applications import resnet
