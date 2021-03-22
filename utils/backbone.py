import glob, os, tarfile, urllib
import tensorflow as tf
from utils import label_map_util


# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def set_model(model_name, label_name):

	# Path to frozen detection graph. This is the actual model that is used for the object detection.
	path_to_pb = os.path.join('custom_frozen_inference_graph', model_name)

	# List of the strings that is used to add correct label for each box.
	path_to_labels = os.path.join('data', label_name)

	num_classes = 36 # no. of id's in labelmap

	# Load a (frozen) Tensorflow model into memory.
	detection_graph = tf.Graph()
	with detection_graph.as_default():
	  od_graph_def = tf.GraphDef()
	  with tf.gfile.GFile(path_to_pb, 'rb') as fid:
	    serialized_graph = fid.read()
	    od_graph_def.ParseFromString(serialized_graph)
	    tf.import_graph_def(od_graph_def, name='')

	# Loading label map
	# Label maps map indices to category names, so that when our convolution network predicts 5,
	# we know that this corresponds to airplane. Here I
	# use internal utility functions, but anything that returns a dictionary
	# mapping integers to appropriate string labels would be fine
	label_map = label_map_util.load_labelmap(path_to_labels)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)

	return detection_graph, category_index

def set_model_classification(model_name, label_name):

	# Path to frozen detection graph. This is the actual model that is used for the object classification.
	path_to_pb = os.path.join('custom_frozen_inference_graph', model_name)

	# List of the strings that is used to add correct label for each box.
	path_to_labels = os.path.join('data', label_name)

	# Loading label map
	# Label maps map indices to category names, so that when our convolution network predicts 5,
	# we know that this corresponds to airplane. Here I
	# use internal utility functions, but anything that returns a dictionary
	# mapping integers to appropriate string labels would be fine
	labels_ = [line.rstrip() for line in tf.gfile.GFile(path_to_labels)]

	# Load a (frozen) Tensorflow model into memory.
	detection_graph = tf.Graph()
	with detection_graph.as_default():
	  od_graph_def = tf.GraphDef()
	  with tf.gfile.GFile(path_to_pb, 'rb') as fid:
	    serialized_graph = fid.read()
	    od_graph_def.ParseFromString(serialized_graph)
	    tf.import_graph_def(od_graph_def, name='')

	return detection_graph, labels_