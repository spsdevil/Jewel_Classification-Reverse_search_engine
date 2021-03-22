import os
import cv2
import time
import random
import pickle
import scipy.io
import numpy as np
import tensorflow as tf

from PIL import Image
from scipy import ndimage
from database import database
from scipy.misc import imsave
from tempfile import TemporaryFile
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

def intersectionR(lst1, lst2, lst3): 
    lst4 = [value for value in lst1 if value in lst2 and value in lst3] 
    return lst4

def intersection(lst1, lst2): 
    lst4 = [value for value in lst1 if value in lst2] 
    return lst4

def get_top_k_similar_ring(image_data_d, image_data_d_c, image_data_d_m, pred_d, pred_d_c, pred_d_m, pred_final, filename, confidence):
  # print("total data",len(pred_final))
  # cosine calculates the cosine distance, not similiarity. Hence no need to reverse list
  cosine_similar_d = [cosine(image_data_d, pred_row) for ith_row, pred_row in enumerate(pred_d)]
  cosine_similar_d_c = [cosine(image_data_d_c, pred_row) for ith_row, pred_row in enumerate(pred_d_c)]
  cosine_similar_d_m = [cosine(image_data_d_m, pred_row) for ith_row, pred_row in enumerate(pred_d_m)]
  top_k_ind_d = np.argsort(cosine_similar_d)
  top_k_ind_d_c = np.argsort(cosine_similar_d_c)
  top_k_ind_d_m = np.argsort(cosine_similar_d_m)

  # DESIGN RESULTS-------------------------
  result_image_d = {}
  for neighbor in top_k_ind_d:
    cosine_value = cosine_similar_d[neighbor]
    similarties = (1 - cosine_value)*100
    if similarties >=  confidence:
      data_new = pred_final[neighbor]#.split('/')[-1]
      result_image_d[data_new] = similarties
  print("Length of RESULT DESIGN  :  ", len(result_image_d))

  # DESIGN_CENTRE RESULTS-------------------
  result_image_d_c = {}
  for neighbor in top_k_ind_d_c:
    cosine_value = cosine_similar_d_c[neighbor]
    similarties = (1 - cosine_value)*100
    if similarties >=  confidence:
      data_new = pred_final[neighbor]#.split('/')[-1]
      result_image_d_c[data_new] = similarties
  print("Length of RESULT D_CENTRE:  ", len(result_image_d_c))

  # DESIGN_METAL RESULTS-------------------
  result_image_d_m = {}
  for neighbor in top_k_ind_d_m:
    cosine_value = cosine_similar_d_m[neighbor]
    similarties = (1 - cosine_value)*100
    if similarties >=  confidence:
      data_new = pred_final[neighbor]#.split('/')[-1]
      result_image_d_m[data_new] = similarties
  print("Length of RESULT D_METAL :  ", len(result_image_d_m))

  # LIST OF ALL FILES(KEYS OF DICT)----------------------
  result_d   = result_image_d.keys()
  result_d_c = result_image_d_c.keys()
  result_d_m = result_image_d_m.keys()

  temp_result = intersectionR(result_d, result_d_c, result_d_m)
  print("final_result: ",len(temp_result))
  final_result = []
  for file in temp_result:
    similarties_d   = result_image_d[file]
    similarties_d_c = result_image_d_c[file]
    similarties_d_m = result_image_d_m[file]
    similarties_avg = (similarties_d+similarties_d_c+similarties_d_m)/3
    final_result.append((similarties_avg, file))#.split('/')[-1]))
    final_result.sort(key = lambda x: x[0], reverse = True)
  return final_result,len(temp_result),len(pred_final)

def get_top_k_similar_necklace(image_data_d, pred_d, pred_final, filename, confidence):
  # print("total data",len(pred_final))
  # cosine calculates the cosine distance, not similiarity. Hence no need to reverse list
  cosine_similar_d = [cosine(image_data_d, pred_row) for ith_row, pred_row in enumerate(pred_d)]
  top_k_ind_d = np.argsort(cosine_similar_d)

  # DESIGN RESULTS-------------------------
  result_image_d = {}
  for neighbor in top_k_ind_d:
    cosine_value = cosine_similar_d[neighbor]
    similarties = (1 - cosine_value)*100
    if similarties >=  confidence:
      data_new = pred_final[neighbor]#.split('/')[-1]
      result_image_d[data_new] = similarties
  print("Length of RESULT DESIGN  :  ", len(result_image_d))

  # LIST OF ALL FILES(KEYS OF DICT)----------------------
  result_d   = result_image_d.keys()

  temp_result = result_d
  print("final_result: ",len(temp_result))
  final_result = []
  for file in temp_result:
    similarties_d   = result_image_d[file]
    similarties_avg = (similarties_d)
    final_result.append((similarties_avg, file))#.split('/')[-1]))
    final_result.sort(key = lambda x: x[0], reverse = True)
  return final_result,len(temp_result),len(pred_final)

def get_top_k_similar_earring(image_data_d, pred_d, pred_final, filename, confidence):
  print("total data",len(pred_final))
  # cosine calculates the cosine distance, not similiarity. Hence no need to reverse list
  cosine_similar_d = [cosine(image_data_d, pred_row) for ith_row, pred_row in enumerate(pred_d)]
  top_k_ind_d = np.argsort(cosine_similar_d)

  # DESIGN RESULTS-------------------------
  result_image_d = {}
  for neighbor in top_k_ind_d:
    cosine_value = cosine_similar_d[neighbor]
    similarties = (1 - cosine_value)*100
    if similarties >=  confidence:
      data_new = pred_final[neighbor]#.split('/')[-1]
      result_image_d[data_new] = similarties
  print("Length of RESULT DESIGN  :  ", len(result_image_d))

  # LIST OF ALL FILES(KEYS OF DICT)----------------------
  result_d   = result_image_d.keys()

  temp_result = result_d
  print("final_result: ",len(temp_result))
  final_result = []
  for file in temp_result:
    similarties_d   = result_image_d[file]
    similarties_avg = (similarties_d)
    final_result.append((similarties_avg, file))#.split('/')[-1]))
    final_result.sort(key = lambda x: x[0], reverse = True)
  return final_result,len(temp_result),len(pred_final)

def get_top_k_similar_pendant(image_data_d, image_data_d_c, pred_d, pred_d_c, pred_final, filename, confidence):
  # cosine calculates the cosine distance, not similiarity. Hence no need to reverse list
  cosine_similar_d = [cosine(image_data_d, pred_row) for ith_row, pred_row in enumerate(pred_d)]
  cosine_similar_d_c = [cosine(image_data_d_c, pred_row) for ith_row, pred_row in enumerate(pred_d_c)]
  top_k_ind_d = np.argsort(cosine_similar_d)
  top_k_ind_d_c = np.argsort(cosine_similar_d_c)

  # DESIGN RESULTS-------------------------
  result_image_d = {}
  for neighbor in top_k_ind_d:
    cosine_value = cosine_similar_d[neighbor]
    similarties = (1 - cosine_value)*100
    if similarties >=  confidence:
      data_new = pred_final[neighbor]#.split('/')[-1]
      result_image_d[data_new] = similarties
  print("Length of RESULT DESIGN  :  ", len(result_image_d))

  # DESIGN_CENTRE RESULTS-------------------
  result_image_d_c = {}
  for neighbor in top_k_ind_d_c:
    cosine_value = cosine_similar_d_c[neighbor]
    similarties = (1 - cosine_value)*100
    if similarties >=  confidence:
      data_new = pred_final[neighbor]#.split('/')[-1]
      result_image_d_c[data_new] = similarties
  print("Length of RESULT D_CENTRE:  ", len(result_image_d_c))

  # LIST OF ALL FILES(KEYS OF DICT)----------------------
  result_d   = result_image_d.keys()
  result_d_c = result_image_d_c.keys()

  temp_result = intersection(result_d, result_d_c)
  print("final_result: ",len(temp_result))
  final_result = []
  for file in temp_result:
    similarties_d   = result_image_d[file]
    similarties_d_c = result_image_d_c[file]
    similarties_avg = (similarties_d+similarties_d_c)/2
    final_result.append((similarties_avg, file))#.split('/')[-1]))
    final_result.sort(key = lambda x: x[0], reverse = True)
  return final_result,len(temp_result),len(pred_final)

def get_top_k_similar_bracelet(image_data_d, image_data_d_c, pred_d, pred_d_c, pred_final, filename, confidence):
  # cosine calculates the cosine distance, not similiarity. Hence no need to reverse list
  cosine_similar_d = [cosine(image_data_d, pred_row) for ith_row, pred_row in enumerate(pred_d)]
  cosine_similar_d_c = [cosine(image_data_d_c, pred_row) for ith_row, pred_row in enumerate(pred_d_c)]
  top_k_ind_d = np.argsort(cosine_similar_d)
  top_k_ind_d_c = np.argsort(cosine_similar_d_c)

  # DESIGN RESULTS-------------------------
  result_image_d = {}
  for neighbor in top_k_ind_d:
    cosine_value = cosine_similar_d[neighbor]
    similarties = (1 - cosine_value)*100
    if similarties >=  confidence:
      data_new = pred_final[neighbor]#.split('/')[-1]
      result_image_d[data_new] = similarties
  print("Length of RESULT DESIGN  :  ", len(result_image_d))

  # DESIGN_CENTRE RESULTS-------------------
  result_image_d_c = {}
  for neighbor in top_k_ind_d_c:
    cosine_value = cosine_similar_d_c[neighbor]
    similarties = (1 - cosine_value)*100
    if similarties >=  confidence:
      data_new = pred_final[neighbor]#.split('/')[-1]
      result_image_d_c[data_new] = similarties
  print("Length of RESULT D_CENTRE:  ", len(result_image_d_c))

  # LIST OF ALL FILES(KEYS OF DICT)----------------------
  result_d   = result_image_d.keys()
  result_d_c = result_image_d_c.keys()

  temp_result = intersection(result_d, result_d_c)
  print("final_result: ",len(temp_result))
  final_result = []
  for file in temp_result:
    similarties_d   = result_image_d[file]
    similarties_d_c = result_image_d_c[file]
    similarties_avg = (similarties_d+similarties_d_c)/2
    final_result.append((similarties_avg, file))#.split('/')[-1]))
    final_result.sort(key = lambda x: x[0], reverse = True)
  return final_result,len(temp_result),len(pred_final)

def create_inception_graph():
  """"Creates a graph from saved GraphDef file and returns a Graph object.

  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
  with tf.Session() as sess:
    model_filename = os.path.join(
        'imagenet', 'classify_image_graph_def.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
          tf.import_graph_def(graph_def, name='', return_elements=[
              BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
              RESIZED_INPUT_TENSOR_NAME]))
  return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

def run_bottleneck_on_image(sess, image_data, image_data_tensor,bottleneck_tensor):
    bottleneck_values = sess.run(
          bottleneck_tensor,
          {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values        

def recommend_ring(imagePath, extracted_features_design, extracted_features_d_c, extracted_features_d_m, neighbors_file, filename, confidence, tensor_tup, category_index):
  tf.reset_default_graph()
  config = tf.ConfigProto(device_count = {'GPU': 0})
  sess = tf.Session(config=config)
  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph())

  Design, Design_centre, Design_metal = object_detection_api.design_extraction_ring(tensor_tup, category_index, imagePath)
  features_d = run_bottleneck_on_image(sess, Design, jpeg_data_tensor, bottleneck_tensor)
  features_d_c = run_bottleneck_on_image(sess, Design_centre, jpeg_data_tensor, bottleneck_tensor)
  features_d_m = run_bottleneck_on_image(sess, Design_metal, jpeg_data_tensor, bottleneck_tensor)

  with open('vectors/{}.pickle'.format(neighbors_file),'rb') as f:
    neighbor_list = pickle.load(f)
  print("loaded images")
  result_image, total_result,total_images = get_top_k_similar_ring(features_d, features_d_c, features_d_m, extracted_features_design, \
                                  extracted_features_d_c, extracted_features_d_m, neighbor_list, filename, confidence)
  database.db_insert_query(filename,total_result,total_images)
  database.db_insert_result(filename, result_image)

  return result_image, total_result, total_images

def recommend_pendant(imagePath, extracted_features_design, extracted_features_d_c, neighbors_file, filename, confidence, tensor_tup, category_index):
  tf.reset_default_graph()
  config = tf.ConfigProto(device_count = {'GPU': 0})
  sess = tf.Session(config=config)
  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph())

  Design, Design_centre = object_detection_api.design_extraction_pendant(tensor_tup, category_index, imagePath)
  features_d = run_bottleneck_on_image(sess, Design, jpeg_data_tensor, bottleneck_tensor)
  features_d_c = run_bottleneck_on_image(sess, Design_centre, jpeg_data_tensor, bottleneck_tensor)

  with open('vectors/{}.pickle'.format(neighbors_file),'rb') as f:
    neighbor_list = pickle.load(f)
  print("loaded images")
  result_image, total_result,total_images = get_top_k_similar_pendant(features_d, features_d_c, extracted_features_design, \
                                  extracted_features_d_c, neighbor_list, filename, confidence)
  database.db_insert_query(filename,total_result,total_images)
  database.db_insert_result(filename, result_image)

  return result_image, total_result, total_images

def recommend_bracelet(imagePath, extracted_features_design, extracted_features_d_c, neighbors_file, filename, confidence, tensor_tup, category_index):
  tf.reset_default_graph()
  config = tf.ConfigProto(device_count = {'GPU': 0})
  sess = tf.Session(config=config)
  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph())

  Design, Design_centre = object_detection_api.design_extraction_bracelate(tensor_tup, category_index, imagePath)
  features_d = run_bottleneck_on_image(sess, Design, jpeg_data_tensor, bottleneck_tensor)
  features_d_c = run_bottleneck_on_image(sess, Design_centre, jpeg_data_tensor, bottleneck_tensor)

  with open('vectors/{}.pickle'.format(neighbors_file),'rb') as f:
    neighbor_list = pickle.load(f)
  print("loaded images")
  result_image, total_result,total_images = get_top_k_similar_bracelet(features_d, features_d_c, extracted_features_design, \
                                  extracted_features_d_c, neighbor_list, filename, confidence)
  database.db_insert_query(filename,total_result,total_images)
  database.db_insert_result(filename, result_image)

  return result_image, total_result, total_images

def recommend_necklace(imagePath, extracted_features_design, neighbors_file, filename, confidence, tensor_tup, category_index):
  tf.reset_default_graph()
  config = tf.ConfigProto(device_count = {'GPU': 0})
  sess = tf.Session(config=config)
  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph())

  Design = object_detection_api.design_extraction_necklace(tensor_tup, category_index, imagePath)
  features_d = run_bottleneck_on_image(sess, Design, jpeg_data_tensor, bottleneck_tensor)

  with open('vectors/{}.pickle'.format(neighbors_file),'rb') as f:
    neighbor_list = pickle.load(f)
  print("loaded images")
  result_image, total_result,total_images = get_top_k_similar_necklace(features_d, extracted_features_design, neighbor_list, filename, confidence)
  database.db_insert_query(filename,total_result,total_images)
  database.db_insert_result(filename, result_image)

  return result_image, total_result, total_images

def recommend_earring(imagePath, extracted_features_design, neighbors_file, filename, confidence, tensor_tup, category_index):
  tf.reset_default_graph()
  config = tf.ConfigProto(device_count = {'GPU': 0})
  sess = tf.Session(config=config)
  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph())

  Design = object_detection_api.design_extraction_earring(tensor_tup, category_index, imagePath)
  features_d = run_bottleneck_on_image(sess, Design, jpeg_data_tensor, bottleneck_tensor)

  with open('vectors/{}.pickle'.format(neighbors_file),'rb') as f:
    neighbor_list = pickle.load(f)
  print("loaded images")
  result_image, total_result,total_images = get_top_k_similar_earring(features_d, extracted_features_design, neighbor_list, filename, confidence)
  database.db_insert_query(filename,total_result,total_images)
  database.db_insert_result(filename, result_image)

  return result_image, total_result, total_images