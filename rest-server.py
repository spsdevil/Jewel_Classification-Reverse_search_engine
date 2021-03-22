# Imports
import warnings
warnings.filterwarnings("ignore")
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import os
import sys
import shutil
import itertools
import numpy as np
import tensorflow as tf

from scipy import ndimage
from scipy.misc import imsave
from database import database
from PIL import Image
from io import BytesIO
from flask_httpauth import HTTPBasicAuth
from werkzeug.utils import secure_filename
from tensorflow.python.platform import gfile
from flask import Flask, jsonify, abort, request, make_response, url_for,redirect, render_template

from lib import search
from api import object_detection_api, object_classification_api
from utils import backbone, image_vectorizer

import tensorflow.compat.v1 as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app = Flask(__name__, static_url_path = "")
auth = HTTPBasicAuth()
######################################################## INITIALIZE DB ###################################################################
con = database.connection()

################################################ Loading the model for object detection ##################################################

detection_graph_Necklace_D, category_index_Necklace_D = backbone.set_model('frozen_inference_graph_Necklace_D.pb', 'label_map_Necklace_D.pbtxt')
sess_n = tf.Session(graph=detection_graph_Necklace_D)
# Definite input and output Tensors for detection_graph
image_tensor_n = detection_graph_Necklace_D.get_tensor_by_name('image_tensor:0')
detection_boxes_n = detection_graph_Necklace_D.get_tensor_by_name('detection_boxes:0')
detection_scores_n = detection_graph_Necklace_D.get_tensor_by_name('detection_scores:0')
detection_classes_n = detection_graph_Necklace_D.get_tensor_by_name('detection_classes:0')
num_detections_n = detection_graph_Necklace_D.get_tensor_by_name('num_detections:0')
necklace_ = (sess_n,image_tensor_n,detection_boxes_n,detection_scores_n,detection_classes_n,num_detections_n)

detection_graph_Bracelet_D_C, category_index_Bracelet_D_C = backbone.set_model('frozen_inference_graph_Bracelet_D_C.pb', 'label_map_Bracelet_D_C.pbtxt')
sess_b = tf.Session(graph=detection_graph_Bracelet_D_C)
# Definite input and output Tensors for detection_graph
image_tensor_b = detection_graph_Bracelet_D_C.get_tensor_by_name('image_tensor:0')
detection_boxes_b = detection_graph_Bracelet_D_C.get_tensor_by_name('detection_boxes:0')
detection_scores_b = detection_graph_Bracelet_D_C.get_tensor_by_name('detection_scores:0')
detection_classes_b = detection_graph_Bracelet_D_C.get_tensor_by_name('detection_classes:0')
num_detections_b = detection_graph_Bracelet_D_C.get_tensor_by_name('num_detections:0')
bracelet_ = (sess_b,image_tensor_b,detection_boxes_b,detection_scores_b,detection_classes_b,num_detections_b)

detection_graph_Ring_D_C_M, category_index_Ring_D_C_M = backbone.set_model('frozen_inference_graph_Ring_D_C_M.pb', 'label_map_Ring_D_C_M.pbtxt')
sess_r = tf.Session(graph=detection_graph_Ring_D_C_M)
# Definite input and output Tensors for detection_graph
image_tensor_r = detection_graph_Ring_D_C_M.get_tensor_by_name('image_tensor:0')
detection_boxes_r = detection_graph_Ring_D_C_M.get_tensor_by_name('detection_boxes:0')
detection_scores_r = detection_graph_Ring_D_C_M.get_tensor_by_name('detection_scores:0')
detection_classes_r = detection_graph_Ring_D_C_M.get_tensor_by_name('detection_classes:0')
num_detections_r = detection_graph_Ring_D_C_M.get_tensor_by_name('num_detections:0')
ring_ = (sess_r,image_tensor_r,detection_boxes_r,detection_scores_r,detection_classes_r,num_detections_r)

detection_graph_Pendant_D_C, category_index_Pendant_D_C = backbone.set_model('frozen_inference_graph_Pendant_D_C.pb', 'label_map_Pendant_D_C.pbtxt')
sess_p = tf.Session(graph=detection_graph_Pendant_D_C)
# Definite input and output Tensors for detection_graph
image_tensor_p = detection_graph_Pendant_D_C.get_tensor_by_name('image_tensor:0')
detection_boxes_p = detection_graph_Pendant_D_C.get_tensor_by_name('detection_boxes:0')
detection_scores_p = detection_graph_Pendant_D_C.get_tensor_by_name('detection_scores:0')
detection_classes_p = detection_graph_Pendant_D_C.get_tensor_by_name('detection_classes:0')
num_detections_p = detection_graph_Pendant_D_C.get_tensor_by_name('num_detections:0')
pendant_ = (sess_p,image_tensor_p,detection_boxes_p,detection_scores_p,detection_classes_p,num_detections_p)

detection_graph_Earring_D, category_index_Earring_D = backbone.set_model('frozen_inference_graph_Earring_D.pb', 'label_map_Earring_D.pbtxt')
sess_e = tf.Session(graph=detection_graph_Earring_D)
# Definite input and output Tensors for detection_graph
image_tensor_e = detection_graph_Earring_D.get_tensor_by_name('image_tensor:0')
detection_boxes_e = detection_graph_Earring_D.get_tensor_by_name('detection_boxes:0')
detection_scores_e = detection_graph_Earring_D.get_tensor_by_name('detection_scores:0')
detection_classes_e = detection_graph_Earring_D.get_tensor_by_name('detection_classes:0')
num_detections_e = detection_graph_Earring_D.get_tensor_by_name('num_detections:0')
earring_ = (sess_e,image_tensor_e,detection_boxes_e,detection_scores_e,detection_classes_e,num_detections_e)
print("MODEL LOADED SUCCESSFULLY")

# Classification ------
# detection_graph_classification, classification_labels = backbone.set_model_classification('classification_model.pb', 'classification_labels.txt')

######################################################### Model Loaded ###################################################################

#################################### Loading the extracted feature vectors for image retrieval ###########################################

# all Images features ---------------------------------------------- (for searching on all catagories)
# with open('vectors/all_features_recom.txt') as x:
#   temp_1 = []
#   for line in x:
#     temp_1.append(line)
#   num = len(temp_1)
# all_extracted_features=np.zeros((num,2048),dtype=np.float32)
# with open('vectors/all_features_recom.txt') as f:
#         for i,line in enumerate(f):
#             all_extracted_features[i,:]=line.split()
# print("ALL features Loaded")

# Bracelate Images features --------------------------------------------- (for searching on Bracelates)
with open('vectors/bracelate_features_recom_DESIGN.txt') as x:
  temp_1 = []
  for line in x:
    temp_1.append(line)
  num = len(temp_1)
bracelate_extracted_features_design=np.zeros((num,2048),dtype=np.float32)
with open('vectors/bracelate_features_recom_DESIGN.txt') as f:
            for i,line in enumerate(f):
                bracelate_extracted_features_design[i,:]=line.split()
print("Bracelate Design features Loaded")

with open('vectors/bracelate_features_recom_DESIGN_CENTER.txt') as x:
  temp_1 = []
  for line in x:
    temp_1.append(line)
  num = len(temp_1)
bracelate_extracted_features_d_c=np.zeros((num,2048),dtype=np.float32)
with open('vectors/bracelate_features_recom_DESIGN_CENTER.txt') as f:
            for i,line in enumerate(f):
                bracelate_extracted_features_d_c[i,:]=line.split()
print("Bracelate Design_Centre features Loaded")

# Earings Images features ----------------------------------------------- (for searching on Earings)
with open('vectors/earring_features_recom_DESIGN.txt') as x:
  temp_1 = []
  for line in x:
    temp_1.append(line)
  num = len(temp_1)
earring_extracted_features_design=np.zeros((num,2048),dtype=np.float32)
with open('vectors/earring_features_recom_DESIGN.txt') as f:
            for i,line in enumerate(f):
                earring_extracted_features_design[i,:]=line.split()
print("Earings Design features Loaded")

# Necklace Images features ---------------------------------------------- (for searching on Necklace)
with open('vectors/necklace_features_recom_DESIGN.txt') as x:
  temp_1 = []
  for line in x:
    temp_1.append(line)
  num = len(temp_1)
necklace_extracted_features_design=np.zeros((num,2048),dtype=np.float32)
with open('vectors/necklace_features_recom_DESIGN.txt') as f:
            for i,line in enumerate(f):
                necklace_extracted_features_design[i,:]=line.split()
print("Necklace Design features Loaded")

# Pendant_design Images features ------------------------------------------------- (for searching on Pendants)
with open('vectors/pendant_features_recom_DESIGN.txt') as x:
  temp_1 = []
  for line in x:
    temp_1.append(line)
  num = len(temp_1)
pendant_extracted_features_design=np.zeros((num,2048),dtype=np.float32)
with open('vectors/pendant_features_recom_DESIGN.txt') as f:
            for i,line in enumerate(f):
                pendant_extracted_features_design[i,:]=line.split()
print("Pendants design features Loaded")

# Pendant_design_centre Images features ------------------------------------------------- (for searching on Pendants)
with open('vectors/pendant_features_recom_DESIGN_CENTER.txt') as x:
  temp_1 = []
  for line in x:
    temp_1.append(line)
  num = len(temp_1)
pendant_extracted_features_d_c=np.zeros((num,2048),dtype=np.float32)
with open('vectors/pendant_features_recom_DESIGN_CENTER.txt') as f:
            for i,line in enumerate(f):
                pendant_extracted_features_d_c[i,:]=line.split()
print("Pendants design centre features Loaded")

# Rings_design Images features ------------------------------------------------------- (for searching on Rings)
with open('vectors/ring_features_recom_DESIGN.txt') as x:
  temp_1 = []
  for line in x:
    temp_1.append(line)
  num = len(temp_1)
rings_extracted_features_design=np.zeros((num,2048),dtype=np.float32)
with open('vectors/ring_features_recom_DESIGN.txt') as f:
            for i,line in enumerate(f):
                rings_extracted_features_design[i,:]=line.split()
print("Rings design features Loaded")

# Rings_design_centre Images features ------------------------------------------------- (for searching on Rings)
with open('vectors/ring_features_recom_DESIGN_CENTER.txt') as x:
  temp_1 = []
  for line in x:
    temp_1.append(line)
  num = len(temp_1)
rings_extracted_features_d_c=np.zeros((num,2048),dtype=np.float32)
with open('vectors/ring_features_recom_DESIGN_CENTER.txt') as f:
            for i,line in enumerate(f):
                rings_extracted_features_d_c[i,:]=line.split()
print("Rings design centre features Loaded")

# Rings_design_metal Images features ------------------------------------------------- (for searching on Rings)
with open('vectors/ring_features_recom_DESIGN_METAL.txt') as x:
  temp_1 = []
  for line in x:
    temp_1.append(line)
  num = len(temp_1)
rings_extracted_features_d_m=np.zeros((num,2048),dtype=np.float32)
with open('vectors/ring_features_recom_DESIGN_METAL.txt') as f:
            for i,line in enumerate(f):
                rings_extracted_features_d_m[i,:]=line.split()
print("Rings design metal features Loaded")

######################################################## Features Loading Ended ##########################################################

######################################################## catagorization ##################################################################
@app.route('/categories')
def image_categorization():
    print("\t| Select Input Dir/Folder:-------")
    PATH_TO_FOLDER = input("\t| Enter Input Dir/Folder:- :- ")

    success = object_classification_api.classification(detection_graph_classification, classification_labels, PATH_TO_FOLDER)
    if success == "True":
      return render_template('index.html')
######################################################## catagorization Ended ############################################################
######################################################### Vectors Conversion #############################################################
# ----------- add the path Here(API) ------------------------------
@app.route('/vector')
def image_vectorization():
    # Get this mode from api througn select tag anything you want----------
    mode = int(input("Press 1 for all, 2 for Bracelate, 3 for Earings, 4 for Necklace, 5 for Pendents, 6 for Rings: "))

    # FOR ALL IMAGESS IN SINGLE FILE
    if mode == 1:
        bracelets_files = [
          'Design/Bracelate/' + f
          for 
          f
          in
          os.listdir('Design/Bracelate/')
          ]
        earings_files = [
          'Design/Earings/' + f
          for 
          f
          in
          os.listdir('Design/Earings/')
          ]
        necklace_files = [
          'Design/Necklace/' + f
          for 
          f
          in
          os.listdir('Design/Necklace/')
          ]
        pendents_files = [
          'Design/Pendents/' + f
          for 
          f
          in
          os.listdir('Design/Pendants/')
          ]
        rings_files = [
          'Design/Rings/' + f
          for 
          f
          in
          os.listdir('Design/Rings/')
          ]
        # all_files = bracelets_files + earings_files + necklace_files + pendents_files + rings_files
        all_files = bracelets_files + necklace_files + rings_files
        image_vectorizer.vector_conversion(all_files, 'all_features_recom', 'neighbor_all', detection_graph, category_index, mode)
        return print('all_extracted')

    # FOR BRACELATE IMAGESS IN SINGLE FILE
    elif mode == 2:
        bracelets_files = [
          'Design/Bracelate/' + f
          for 
          f
          in
          os.listdir('Design/Bracelate/')
          ]
        all_files = bracelets_files
        image_vectorizer.vector_conversion(all_files, 'bracelate_features_recom', 'neighbor_bracelate', bracelet_, category_index_Bracelet_D_C, mode)
        return print('bracelate_extracted')

    # FOR EARINGS IMAGESS IN SINGLE FILE
    elif mode == 3:
        earings_files = [
          'Design/Earings/' + f
          for 
          f
          in
          os.listdir('Design/Earings/')
          ]
        all_files = earings_files
        image_vectorizer.vector_conversion(all_files, 'earring_features_recom', 'neighbor_earring', earring_, category_index_Earring_D, mode)

    # FOR NECKLACE IMAGESS IN SINGLE FILE
    elif mode == 4:
        necklace_files = [
          'Design/Necklace/' + f
          for 
          f
          in
          os.listdir('Design/Necklace/')
          ]
        all_files = necklace_files
        image_vectorizer.vector_conversion(all_files, 'necklace_features_recom', 'neighbor_necklace', necklace_, category_index_Necklace_D, mode)
        return print('necklace_extracted')

    # FOR PENDANTS IMAGESS IN SINGLE FILE
    elif mode == 5:
        pendents_files = [
          'Design/Pendents/' + f
          for 
          f
          in
          os.listdir('Design/Pendents/')
          ]
        all_files = pendents_files
        image_vectorizer.vector_conversion(all_files, 'pendant_features_recom', 'neighbor_pendant', pendant_, category_index_Pendant_D_C, mode)

    # FOR RINGS IMAGESS IN SINGLE FILE
    elif mode == 6:
        rings_files = [
          'Design/Rings/' + f
          for 
          f
          in
          os.listdir('Design/Rings/')
          ]
        all_files = rings_files
        image_vectorizer.vector_conversion(all_files, 'ring_features_recom','neighbor_ring', ring_, category_index_Ring_D_C_M, mode)
        return print('rings_extracted')

    ############################### NOT COMPLETED #######################################

###############################  This function is used to do the image search/image retrieval ##########################################
@app.route("/sim_update", methods=['GET','POST'])
def sim_update():
  if request.method == 'GET':
    q_id = request.args.get('q_name')
    r_name = request.args.get('c_name')
    sim = request.args.get('sim')
    database.db_update(q_id, r_name, sim)
    return jsonify(success = "True")
  else:
    return jsonify(success = "False")

@app.route("/delete_img", methods=['GET','POST'])
def del_img():
  if request.method == 'GET':
    q_id = request.args.get('q_name')
    r_name = request.args.get('c_name')
    database.db_delete(q_id, r_name)
    return jsonify(success = "True")
  else:
    return jsonify(success = "False")

@app.route("/get_img_for_pages", methods=['GET','POST'])
def get_img():
  if request.method == 'GET':
    q_id = request.args.get('q_name')
    page_n = request.args.get('page')
    success,rows = database.check_img(q_id)
    result_images = database.fet_img(q_id, page_n)
    final_result_images = tuppling(result_images)
    uploaded_img_path = 'static/Upload' + q_id
    return render_template('index.html',
                       query_path=uploaded_img_path,
                       result_images=final_result_images,
                       total_images=int(rows[1]),
                       total_results=int(rows[0]))

################################################## Main function (HOME PAGE) ###########################################################
@app.route("/", methods=['GET', 'POST'])
def main():
  if request.method == 'POST':
    upload = 'static/Upload'
    if not gfile.Exists(upload):
      os.mkdir(upload)
      print('_________________________upload directory_created________________________________')
    app.config["upload"] = upload

    result = 'static/result'
    if not gfile.Exists(result):
      os.mkdir(result)
    # print(request.method)

    confidence = request.form['conf']
    confidence = int(confidence)
    
    mode = request.form['mod']
    mode = int(mode)

    # check if the post request has the file part
    if 'query_img' not in request.files:
        print('No file part')
        return redirect(request.url)
    file = request.files['query_img']
    print("Given File:- ",file.filename)
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
      print('No selected file')
      return redirect(request.url)
    if file and file.filename.split('.')[-1] in ALLOWED_EXTENSIONS:
      filename = secure_filename(file.filename)
      file.save(os.path.join(app.config['upload'], file.filename))

      inputloc = os.path.join(app.config['upload'], file.filename)

      success,rows = database.check_img(filename)
      if success == True:
        result_images,total_results_img = database.fetch_data(filename)
        final_result_images = result_images
        uploaded_img_path = os.path.join(app.config['upload'], filename)[7:]
        return render_template('index.html',
                       query_path=uploaded_img_path,
                       result_images=final_result_images,
                       total_images=int(rows[1]),
                       total_results=int(rows[0]))
      else:
        if mode == 1:
            _,total_results, total_images = search.recommend(inputloc, all_extracted_features_design, all_extracted_features_d_c,\
                                              'neighbor_all', filename, confidence, detection_graph, category_index)
            result_images,total_results_img = database.fetch_data(filename)
            final_result_images = result_images
            uploaded_img_path = os.path.join(app.config['upload'], filename)[7:]
            return render_template('index.html',
                           query_path=uploaded_img_path,
                           result_images=final_result_images,
                           total_images=total_images,
                           total_results=total_results)
        elif mode == 2:
            _,total_results, total_images = search.recommend_bracelet(inputloc, bracelate_extracted_features_design, bracelate_extracted_features_d_c,\
                                            'neighbor_bracelate', filename,  confidence, bracelet_, category_index_Bracelet_D_C)
            result_images,total_results_img = database.fetch_data(filename)
            final_result_images = result_images
            uploaded_img_path = os.path.join(app.config['upload'], filename)[7:]
            return render_template('index.html',
                         query_path=uploaded_img_path,
                         result_images=final_result_images,
                         total_images=total_images,
                         total_results=total_results)
        elif mode == 3:
            _,total_results, total_images = search.recommend_earring(inputloc, earring_extracted_features_design,\
                                              'neighbor_earring', filename, confidence, earring_, category_index_Earring_D)
            result_images,total_results_img = database.fetch_data(filename)
            final_result_images = result_images
            uploaded_img_path = os.path.join(app.config['upload'], filename)[7:]
            return render_template('index.html',
                           query_path=uploaded_img_path,
                           result_images=final_result_images,
                           total_images=total_images,
                           total_results=total_results)
        elif mode == 4:
            _,total_results, total_images = search.recommend_necklace(inputloc, necklace_extracted_features_design,\
                                              'neighbor_necklace', filename, confidence, necklace_, category_index_Necklace_D)
            result_images,total_results_img = database.fetch_data(filename)
            final_result_images = result_images
            uploaded_img_path = os.path.join(app.config['upload'], filename)[7:]
            return render_template('index.html',
                           query_path=uploaded_img_path,
                           result_images=final_result_images,
                           total_images=total_images,
                           total_results=total_results)
        elif mode == 5:
            _,total_results, total_images = search.recommend_pendant(inputloc, pendant_extracted_features_design, pendant_extracted_features_d_c,\
                                              'neighbor_pendant', filename, confidence, pendant_, category_index_Pendant_D_C)
            result_images,total_results_img = database.fetch_data(filename)
            final_result_images = result_images
            uploaded_img_path = os.path.join(app.config['upload'], filename)[7:]
            return render_template('index.html',
                           query_path=uploaded_img_path,
                           result_images=final_result_images,
                           total_images=total_images,
                           total_results=total_results)
        elif mode == 6:
            _,total_results, total_images = search.recommend_ring(inputloc, rings_extracted_features_design, rings_extracted_features_d_c,\
                                              rings_extracted_features_d_m, 'neighbor_ring', filename, confidence,
                                              ring_, category_index_Ring_D_C_M)
            result_images,total_results_img = database.fetch_data(filename)
            final_result_images = result_images
            uploaded_img_path = os.path.join(app.config['upload'], filename)[7:]
            return render_template('index.html',
                           query_path=uploaded_img_path,
                           result_images=final_result_images,
                           total_images=total_images,
                           total_results=total_results)
  else:
      return render_template('index.html')

@app.route("/Jewel_recommend", methods=['GET', 'POST'])
def Jewel_recommend():
  # initialize the data dictionary to be returned by the request
  data = {"Success": False}
  if request.method == 'POST':
    upload = 'static/Upload'
    if not gfile.Exists(upload):
      os.mkdir(upload)
    app.config["upload"] = upload

    confidence = request.form['conf']
    confidence = int(confidence)
    
    mode = request.form['mode']
    mode = int(mode)

    # check if the post request has the file part
    if 'image' not in request.files:
        print('No file part')
        data.update({"File": "No File Selected"})
        return jsonify(data)

    file = request.files['image']
    print(file.filename)
    filename = file.filename
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
      print('No file part')
      data.update({"File": "No File Selected"})
      return jsonify(data)
    if file and file.filename.split('.')[-1] in ALLOWED_EXTENSIONS:
      inputloc = _grab_image(stream=request.files["image"])
      
      success,rows = database.check_img(file.filename)
      if success == True:
        result_images,total_results_img = database.fetch_data(filename)
        uploaded_img_path = os.path.join(app.config['upload'], filename)[7:]
        data.update({'Success':True, 'Total_images': total_results_img, 'Number_of_results':len(result_images), 'Query_Image': uploaded_img_path,'Results': result_images})
        return jsonify(data)
      else:
        if mode == 1:
            _,total_results, total_images = search.recommend(inputloc, all_extracted_features_design, all_extracted_features_d_c,\
                                              'neighbor_all', filename, confidence, detection_graph, category_index)
            result_images,total_results_img = database.fetch_data(filename)
            uploaded_img_path = os.path.join(app.config['upload'], filename)[7:]
            data.update({'Success':True, 'Total_images': total_images, 'Number_of_results':total_results, 'Query_Image': uploaded_img_path,'Results': result_images})
            return jsonify(data)
        elif mode == 2:
            _,total_results, total_images = search.recommend_bracelet(inputloc, bracelate_extracted_features_design, bracelate_extracted_features_d_c,\
                                            'neighbor_bracelate', filename,  confidence, bracelet_, category_index_Bracelet_D_C)
            result_images,total_results_img = database.fetch_data(filename)
            uploaded_img_path = os.path.join(app.config['upload'], filename)[7:]
            data.update({'Success':True, 'Total_images': total_images, 'Number_of_results':total_results, 'Query_Image': uploaded_img_path,'Results': result_images})
            return jsonify(data)
        elif mode == 3:
            _,total_results, total_images = search.recommend_earring(inputloc, earring_extracted_features_design,\
                                              'neighbor_earring', filename, confidence, earring_, category_index_Earring_D)
            result_images,total_results_img = database.fetch_data(filename)
            uploaded_img_path = os.path.join(app.config['upload'], filename)[7:]
            data.update({'Success':True, 'Total_images': total_images, 'Number_of_results':total_results, 'Query_Image': uploaded_img_path,'Results': result_images})
            return jsonify(data)
        elif mode == 4:
            _,total_results, total_images = search.recommend_necklace(inputloc, necklace_extracted_features_design,\
                                              'neighbor_necklace', filename, confidence, necklace_, category_index_Necklace_D)
            result_images,total_results_img = database.fetch_data(filename)
            uploaded_img_path = os.path.join(app.config['upload'], filename)[7:]
            data.update({'Success':True, 'Total_images': total_images, 'Number_of_results':total_results, 'Query_Image': uploaded_img_path,'Results': result_images})
            return jsonify(data)
        elif mode == 5:
            _,total_results, total_images = search.recommend_pendant(inputloc, pendant_extracted_features_design, pendant_extracted_features_d_c,\
                                              'neighbor_pendant', filename, confidence, pendant_, category_index_Pendant_D_C)
            result_images,total_results_img = database.fetch_data(filename)
            uploaded_img_path = os.path.join(app.config['upload'], filename)[7:]
            data.update({'Success':True, 'Total_images': total_images, 'Number_of_results':total_results, 'Query_Image': uploaded_img_path,'Results': result_images})
            return jsonify(data)
        elif mode == 6:
            _,total_results, total_images = search.recommend_ring(inputloc, rings_extracted_features_design, rings_extracted_features_d_c,\
                                              rings_extracted_features_d_m, 'neighbor_ring', filename, confidence,
                                              ring_, category_index_Ring_D_C_M)
            result_images,total_results_img = database.fetch_data(filename)
            uploaded_img_path = os.path.join(app.config['upload'], filename)[7:]
            data.update({'Success':True, 'Total_images': total_images, 'Number_of_results':total_results, 'Query_Image': uploaded_img_path,'Results': result_images})
            return jsonify(data)

def _grab_image(stream=None, url=None):
    # if the URL is not None, then download the image
    if url is not None:
        resp = urllib.request.urlopen(url)
        data = resp.read()
    # if the stream is not None, then the image has been uploaded
    elif stream is not None:
        data = stream.read()
    unknown_image = Image.open(BytesIO(data))
    upload = 'static/upload'
    app.config["upload"] = upload
    unknown_image.save(os.path.join(app.config['upload'], stream.filename))
    q_img_path = os.path.join(app.config['upload'], stream.filename)
    return q_img_path

if __name__ == '__main__':
    app.run(debug = True, host="192.168.1.14")