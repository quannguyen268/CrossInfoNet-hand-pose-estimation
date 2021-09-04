import tensorflow as tf
tf = tf.compat.v1
tf.disable_v2_behavior()
def printTensors(pb_file):

    # read pb into graph_def
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # print operations
    for op in graph.get_operations():
        print(op.name)


printTensors('/home/quan/PycharmProjects/hand_estimation/model/')

graph = tf.saved_model.load('/home/quan/PycharmProjects/MiAI_FaceRecog_2/Models/20180402-114759.pb')
for op in graph.get_operations():
    abc = graph.get_tensor_by_name(op.name + ":0")
    print(abc)


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('./model.ckpt.meta', input_map=None)
        saver.restore(tf.get_default_session(), './model.ckpt')
        name = [n.name for n in tf.get_default_graph().as_graph_def().node]

        all_nodes = [n for n in tf.get_default_graph().as_graph_def().node]
        all_ops = tf.get_default_graph().get_operations()
        all_vars = tf.global_variables()
        all_tensors = [tensor for op in tf.get_default_graph().get_operations() for tensor in op.values()]
        all_placeholders = [placeholder for op in tf.get_default_graph().get_operations() if op.type=='Placeholder' for placeholder in op.values()]
        for op in tf.get_default_graph().get_operations():
            print(str(op.name))
#%%
outputs = list(map(lambda tname: tf.get_default_graph().get_tensor_by_name(tname), [
    'DCNN/block3_pool/MaxPool:0',
    'DCNN/block4_pool/MaxPool:0',
    'DCNN/block5_pool/MaxPool:0'
]))

with tf.Session() as sess:
    val_outputs = sess.run(outputs)


#%%

### V1
import tensorflow as tf
from tensorflow.python.platform import gfile
GRAPH_PB_PATH = './frozen_model.pb'
with tf.Session() as sess:
   print("load graph")
   with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
       graph_def = tf.GraphDef()
   graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   tf.import_graph_def(graph_def, name='')
   graph_nodes=[n for n in graph_def.node]
   names = []
   for t in graph_nodes:
      names.append(t.name)
   print(names)



### V2

GRAPH_PB_PATH = './frozen_model.pb'
with tf.compat.v1.Session() as sess:
   print("load graph")
   with tf.io.gfile.GFile(GRAPH_PB_PATH,'rb') as f:
       graph_def = tf.compat.v1.GraphDef()
   graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   tf.import_graph_def(graph_def, name='')
   graph_nodes=[n for n in graph_def.node]
   names = []
   for t in graph_nodes:
      names.append(t.name)
   print(names)