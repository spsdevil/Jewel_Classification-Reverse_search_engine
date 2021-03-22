import os
import cv2
import pickle
import random
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import ndimage
from api import object_detection_api
from scipy.spatial.distance import cosine
from tensorflow.python.platform import gfile
from sklearn.neighbors import NearestNeighbors



BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

def create_inception_graph():
  """"Creates a graph from saved GraphDef file and returns a Graph object.
  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
  with tf.Session() as sess:
    model_filename = os.path.join(
        'imagenet', 'classify_image_graph_def.pb')
      # 'imagenet', 'classification_model.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.compat.v1.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
          tf.import_graph_def(graph_def, name='', return_elements=[
              BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
              RESIZED_INPUT_TENSOR_NAME]))
  return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
  bottleneck_values = sess.run(
        bottleneck_tensor,
        {image_data_tensor: image_data})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values

def vector_conversion(all_files, vector_file_name, neighbours_name, tensor_tup, category_index, mode ):
  # Get outputs from second-to-last layer in pre-built model
  print("Total Images: ", len(all_files))

  random.shuffle(all_files)

  # num_images = 10000
  num_images= int(len(all_files))
  neighbor_list = all_files[:num_images]

  with open('vectors/New/{}.pickle'.format(neighbours_name),'wb') as f:
          pickle.dump(neighbor_list,f)
  print("saved neighbour list")

  extracted_features_design = np.ndarray((num_images, 2048))
  extracted_features_D_centre = np.ndarray((num_images, 2048))
  extracted_features_D_metal = np.ndarray((num_images, 2048))
  sess = tf.Session()
  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph()) 


  for i, data in enumerate(tqdm(neighbor_list, desc = neighbours_name.split('_')[-1].title() + ' Vectorization')):
      if data.endswith('.db'):
        os.remove(data)
      elif data.endswith(".jpeg") or data.endswith(".jpg") or data.endswith(".JPG") or data.endswith(".JPEG") \
              or data.endswith(".png") or data.endswith(".PNG") or data.endswith(".bmp") or data.endswith(".BMP"):

          # For ALL ----
          if mode == 1:
            Design, Design_centre, Design_metal = object_detection_api.design_extraction_all(tensor_tup, category_index, data)
            if Design != None and Design_centre != None:
              feature_d = run_bottleneck_on_image(sess, Design, jpeg_data_tensor, bottleneck_tensor)
              extracted_features_design[i:i+1] = feature_d

              feature_d_c = run_bottleneck_on_image(sess, Design_centre, jpeg_data_tensor, bottleneck_tensor)
              extracted_features_D_centre[i:i+1] = feature_d_c
            else:
              pass
          # For BRACELATE ----
          elif mode == 2:
            Design, Design_centre = object_detection_api.design_extraction_bracelate(tensor_tup, category_index, data)
            if Design != None and Design_centre != None:
              feature_d = run_bottleneck_on_image(sess, Design, jpeg_data_tensor, bottleneck_tensor)
              extracted_features_design[i:i+1] = feature_d

              feature_d_c = run_bottleneck_on_image(sess, Design_centre, jpeg_data_tensor, bottleneck_tensor)
              extracted_features_D_centre[i:i+1] = feature_d_c
            else:
              pass
          # For EARING ----
          elif mode == 3:
            Design = object_detection_api.design_extraction_earring(tensor_tup, category_index, data)
            if Design != None:
              feature_d = run_bottleneck_on_image(sess, Design, jpeg_data_tensor, bottleneck_tensor)
              extracted_features_design[i:i+1] = feature_d
            else:
              pass
          # For NECKLACE ----
          elif mode == 4:
            Design = object_detection_api.design_extraction_necklace(tensor_tup, category_index, data)
            if Design != None:
              feature_d = run_bottleneck_on_image(sess, Design, jpeg_data_tensor, bottleneck_tensor)
              extracted_features_design[i:i+1] = feature_d
            else:
              pass
          # For PENDANT ----
          elif mode == 5:
            Design, Design_centre = object_detection_api.design_extraction_pendant(tensor_tup, category_index, data)
            if Design != None and Design_centre != None:
              feature_d = run_bottleneck_on_image(sess, Design, jpeg_data_tensor, bottleneck_tensor)
              extracted_features_design[i:i+1] = feature_d

              feature_d_c = run_bottleneck_on_image(sess, Design_centre, jpeg_data_tensor, bottleneck_tensor)
              extracted_features_D_centre[i:i+1] = feature_d_c
            else:
              pass
          # For RING ----
          elif mode == 6:
            Design, Design_centre, Design_metal = object_detection_api.design_extraction_ring(tensor_tup, category_index, data)
            if Design != None and Design_centre != None and Design_metal != None:
              feature_d = run_bottleneck_on_image(sess, Design, jpeg_data_tensor, bottleneck_tensor)
              extracted_features_design[i:i+1] = feature_d

              feature_d_c = run_bottleneck_on_image(sess, Design_centre, jpeg_data_tensor, bottleneck_tensor)
              extracted_features_D_centre[i:i+1] = feature_d_c

              feature_d_m = run_bottleneck_on_image(sess, Design_metal, jpeg_data_tensor, bottleneck_tensor)
              extracted_features_D_metal[i:i+1] = feature_d_m
            else:
              pass
  if mode == 1:
    np.savetxt("vectors/{}_DESIGN.txt".format(vector_file_name), extracted_features_design)
    np.savetxt("vectors/{}_DESIGN_CENTER.txt".format(vector_file_name), extracted_features_D_centre)
    print("saved exttracted features {}_DESIGN, {}_DESIGN_CENTER". format(vector_file_name,vector_file_name))
  elif mode == 2:
    np.savetxt("vectors/New/{}_DESIGN.txt".format(vector_file_name), extracted_features_design)
    np.savetxt("vectors/New/{}_DESIGN_CENTER.txt".format(vector_file_name), extracted_features_D_centre)
    print("saved exttracted features {}_DESIGN, {}_DESIGN_CENTER". format(vector_file_name,vector_file_name))
  elif mode == 3:
    np.savetxt("vectors/{}_DESIGN.txt".format(vector_file_name), extracted_features_design)
    print("saved exttracted features {}_DESIGN". format(vector_file_name,vector_file_name))
  elif mode == 4:
    np.savetxt("vectors/{}_DESIGN.txt".format(vector_file_name), extracted_features_design)
    print("saved exttracted features {}_DESIGN". format(vector_file_name))
  elif mode == 5:
    np.savetxt("vectors/{}_DESIGN.txt".format(vector_file_name), extracted_features_design)
    np.savetxt("vectors/{}_DESIGN_CENTER.txt".format(vector_file_name), extracted_features_D_centre)
    print("saved exttracted features {}_DESIGN, {}_DESIGN_CENTER". format(vector_file_name,vector_file_name))
  elif mode == 6:
    np.savetxt("vectors/{}_DESIGN.txt".format(vector_file_name), extracted_features_design)
    np.savetxt("vectors/{}_DESIGN_CENTER.txt".format(vector_file_name), extracted_features_D_centre)
    np.savetxt("vectors/{}_DESIGN_METAL.txt".format(vector_file_name), extracted_features_D_metal)
    print("saved exttracted features {}_DESIGN, {}_DESIGN_CENTER and {}_DESIGN_METAL". format(vector_file_name,vector_file_name,vector_file_name))