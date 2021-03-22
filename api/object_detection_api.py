import os
import io
import cv2
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from PIL import Image, ImageOps
from utils import visualization_utils as vis_util

# Design extraction of images present in a dir ------------------------------------
def design_extraction_ring(Tensors_tup, category_index, data):
  # Grab path to current working directory
  CWD_PATH = os.getcwd()

  # Output cropped
  CROP_FOLDER = 'Design'
  PATH_TO_CROP = os.path.join(CWD_PATH, CROP_FOLDER)
  if data.endswith('.db'):
    os.remove(data)
    print("removed")
  elif data.endswith(".jpeg") or data.endswith(".jpg") or data.endswith(".JPG") or data.endswith(".JPEG") \
      or data.endswith(".png") or data.endswith(".PNG") or data.endswith(".bmp") or data.endswith(".BMP"):
    # print("DATA________",data)

    image_str = data.replace('/','\\\\')
    image_str_list = image_str.split('\\\\')
    image_name = image_str_list[-1][:-4]

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(data)
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input

    (boxes, scores, classes, num) = Tensors_tup[0].run(
    [Tensors_tup[2], Tensors_tup[3], Tensors_tup[4], Tensors_tup[5]],
                feed_dict={Tensors_tup[1]: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    _ , _ , present_class = vis_util.visualize_boxes_and_labels_on_image_array(current_frame_number=1,
                                                                              image = image,
                                                                              mode=1,
                                                                              color_recognition_status=0,
                                                                              boxes=np.squeeze(boxes),
                                                                              classes=np.squeeze(classes).astype(np.int64),
                                                                              scores=np.squeeze(scores),
                                                                              category_index=category_index,
                                                                              use_normalized_coordinates=True,
                                                                              line_thickness=8,
                                                                              min_score_thresh=0.70)

    # Detected Objects in image---------------------------------------------------------
    # print(":::::----",present_class.keys())
    # print(":::::---- Imge:- ",image_name," detected label:- ",present_class.keys())

    # Crop the Detected Part from image-------------------------------------------------
    DESIGN = None
    DESIGN_CENTRE = None
    DESIGN_METAL = None
    if "DESIGN" in present_class.keys() and "DESIGN_CENTRE" in present_class.keys() and "DESIGN_METAL" in present_class.keys():
      img = Image.open(data)
      img_grayscale = ImageOps.grayscale(img)
      (frame_height, frame_width) = image.shape[:2]
      ymin, xmin, ymax, xmax = present_class["DESIGN"][1]
      ymin = ymin*frame_height
      xmin = xmin*frame_width
      ymax = ymax*frame_height
      xmax = xmax*frame_width
      cropped_d = img_grayscale.crop((xmin, ymin, xmax, ymax))
      buf = io.BytesIO()
      cropped_d.save(buf, format='JPEG')
      DESIGN = buf.getvalue()

      ymin, xmin, ymax, xmax = present_class["DESIGN_CENTRE"][1]
      ymin = ymin*frame_height
      xmin = xmin*frame_width
      ymax = ymax*frame_height
      xmax = xmax*frame_width
      cropped_d_c = img_grayscale.crop((xmin, ymin, xmax, ymax))
      buf = io.BytesIO()
      cropped_d_c.save(buf, format='JPEG')
      DESIGN_CENTRE = buf.getvalue()

      ymin, xmin, ymax, xmax = present_class["DESIGN_METAL"][1]
      ymin = ymin*frame_height
      xmin = xmin*frame_width
      ymax = ymax*frame_height
      xmax = xmax*frame_width
      cropped_d_m = img_grayscale.crop((xmin, ymin, xmax, ymax))
      buf = io.BytesIO()
      cropped_d_m.save(buf, format='JPEG')
      DESIGN_METAL = buf.getvalue()
    else:
      cv2.imwrite(PATH_TO_CROP + "/The_problematic_one's/Rings/{}.jpg".format(image_name), image)
  return DESIGN, DESIGN_CENTRE, DESIGN_METAL

def design_extraction_pendant(Tensors_tup, category_index, data):
  # Grab path to current working directory
  CWD_PATH = os.getcwd()

  # Output cropped
  CROP_FOLDER = 'Design'
  PATH_TO_CROP = os.path.join(CWD_PATH, CROP_FOLDER)
  if data.endswith('.db'):
    os.remove(data)
    print("removed")
  elif data.endswith(".jpeg") or data.endswith(".jpg") or data.endswith(".JPG") or data.endswith(".JPEG") \
      or data.endswith(".png") or data.endswith(".PNG") or data.endswith(".bmp") or data.endswith(".BMP"):
    # print("DATA________",data)

    image_str = data.replace('/','\\\\')
    image_str_list = image_str.split('\\\\')
    image_name = image_str_list[-1][:-4]

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(data)
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input

    (boxes, scores, classes, num) = Tensors_tup[0].run(
    [Tensors_tup[2], Tensors_tup[3], Tensors_tup[4], Tensors_tup[5]],
                feed_dict={Tensors_tup[1]: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    _ , _ , present_class = vis_util.visualize_boxes_and_labels_on_image_array(current_frame_number=1,
                                                                              image = image,
                                                                              mode=1,
                                                                              color_recognition_status=0,
                                                                              boxes=np.squeeze(boxes),
                                                                              classes=np.squeeze(classes).astype(np.int64),
                                                                              scores=np.squeeze(scores),
                                                                              category_index=category_index,
                                                                              use_normalized_coordinates=True,
                                                                              line_thickness=8,
                                                                              min_score_thresh=0.70)

    # Detected Objects in image---------------------------------------------------------
    # print(":::::----",present_class.keys())
    # print(":::::---- Imge:- ",image_name," detected label:- ",present_class.keys())

    # Crop the Detected Part from image-------------------------------------------------
    DESIGN = None
    DESIGN_CENTRE = None
    if "DESIGN" in present_class.keys() and "DESIGN_CENTRE" in present_class.keys():
      img = Image.open(data)
      img_grayscale = ImageOps.grayscale(img)
      (frame_height, frame_width) = image.shape[:2]
      ymin, xmin, ymax, xmax = present_class["DESIGN"][1]
      ymin = ymin*frame_height
      xmin = xmin*frame_width
      ymax = ymax*frame_height
      xmax = xmax*frame_width
      cropped_d = img_grayscale.crop((xmin, ymin, xmax, ymax))
      buf = io.BytesIO()
      cropped_d.save(buf, format='JPEG')
      DESIGN = buf.getvalue()

      ymin, xmin, ymax, xmax = present_class["DESIGN_CENTRE"][1]
      ymin = ymin*frame_height
      xmin = xmin*frame_width
      ymax = ymax*frame_height
      xmax = xmax*frame_width
      cropped_d_c = img_grayscale.crop((xmin, ymin, xmax, ymax))
      buf = io.BytesIO()
      cropped_d_c.save(buf, format='JPEG')
      DESIGN_CENTRE = buf.getvalue()
    else:
      cv2.imwrite(PATH_TO_CROP + "/The_problematic_one's/Pendents/{}.jpg".format(image_name), image)
    return DESIGN, DESIGN_CENTRE

def design_extraction_earring(Tensors_tup, category_index, data):
  # Grab path to current working directory
  CWD_PATH = os.getcwd()

  # Output cropped
  CROP_FOLDER = 'Design'
  PATH_TO_CROP = os.path.join(CWD_PATH, CROP_FOLDER)
  if data.endswith('.db'):
    os.remove(data)
    print("removed")
  elif data.endswith(".jpeg") or data.endswith(".jpg") or data.endswith(".JPG") or data.endswith(".JPEG") \
      or data.endswith(".png") or data.endswith(".PNG") or data.endswith(".bmp") or data.endswith(".BMP"):
    # print("DATA________",data)

    image_str = data.replace('/','\\\\')
    image_str_list = image_str.split('\\\\')
    image_name = image_str_list[-1][:-4]

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(data)
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input

    (boxes, scores, classes, num) = Tensors_tup[0].run(
    [Tensors_tup[2], Tensors_tup[3], Tensors_tup[4], Tensors_tup[5]],
                feed_dict={Tensors_tup[1]: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    _ , _ , present_class = vis_util.visualize_boxes_and_labels_on_image_array(current_frame_number=1,
                                                                              image = image,
                                                                              mode=1,
                                                                              color_recognition_status=0,
                                                                              boxes=np.squeeze(boxes),
                                                                              classes=np.squeeze(classes).astype(np.int64),
                                                                              scores=np.squeeze(scores),
                                                                              category_index=category_index,
                                                                              use_normalized_coordinates=True,
                                                                              line_thickness=8,
                                                                              min_score_thresh=0.70)

    # Detected Objects in image---------------------------------------------------------
    # print(":::::----",present_class.keys())
    # print(":::::---- Imge:- ",image_name," detected label:- ",present_class.keys())

    # Crop the Detected Part from image-------------------------------------------------
    DESIGN = None
    DESIGN_CENTRE = None
    if "DESIGN" in present_class.keys():
      img = Image.open(data)
      img_grayscale = ImageOps.grayscale(img)
      (frame_height, frame_width) = image.shape[:2]
      ymin, xmin, ymax, xmax = present_class["DESIGN"][1]
      ymin = ymin*frame_height
      xmin = xmin*frame_width
      ymax = ymax*frame_height
      xmax = xmax*frame_width
      cropped_d = img_grayscale.crop((xmin, ymin, xmax, ymax))
      buf = io.BytesIO()
      cropped_d.save(buf, format='JPEG')
      DESIGN = buf.getvalue()  
    else:
      cv2.imwrite(PATH_TO_CROP + "/The_problematic_one's/Earings/{}.jpg".format(image_name), image)
    return DESIGN

def design_extraction_bracelate(Tensors_tup, category_index, data):
  # Grab path to current working directory
  CWD_PATH = os.getcwd()

  # Output cropped
  CROP_FOLDER = 'Design'
  PATH_TO_CROP = os.path.join(CWD_PATH, CROP_FOLDER)
  if data.endswith('.db'):
    os.remove(data)
    print("removed")
  elif data.endswith(".jpeg") or data.endswith(".jpg") or data.endswith(".JPG") or data.endswith(".JPEG") \
      or data.endswith(".png") or data.endswith(".PNG") or data.endswith(".bmp") or data.endswith(".BMP"):
    # print("DATA________",data)

    image_str = data.replace('/','\\\\')
    image_str_list = image_str.split('\\\\')
    image_name = image_str_list[-1][:-4]

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(data)
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input

    (boxes, scores, classes, num) = Tensors_tup[0].run(
    [Tensors_tup[2], Tensors_tup[3], Tensors_tup[4], Tensors_tup[5]],
                feed_dict={Tensors_tup[1]: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    _ , _ , present_class = vis_util.visualize_boxes_and_labels_on_image_array(current_frame_number=1,
                                                                              image = image,
                                                                              mode=1,
                                                                              color_recognition_status=0,
                                                                              boxes=np.squeeze(boxes),
                                                                              classes=np.squeeze(classes).astype(np.int64),
                                                                              scores=np.squeeze(scores),
                                                                              category_index=category_index,
                                                                              use_normalized_coordinates=True,
                                                                              line_thickness=8,
                                                                              min_score_thresh=0.70)

    # Detected Objects in image---------------------------------------------------------
    # print(":::::----",present_class.keys())
    # print(":::::---- Imge:- ",image_name," detected label:- ",present_class.keys())

    # Crop the Detected Part from image-------------------------------------------------
    DESIGN = None
    DESIGN_CENTRE = None
    if "DESIGN" in present_class.keys() and "DESIGN_CENTRE" in present_class.keys():
      img = Image.open(data)
      img_grayscale = ImageOps.grayscale(img)
      (frame_height, frame_width) = image.shape[:2]
      ymin, xmin, ymax, xmax = present_class["DESIGN"][1]
      ymin = ymin*frame_height
      xmin = xmin*frame_width
      ymax = ymax*frame_height
      xmax = xmax*frame_width
      cropped_d = img_grayscale.crop((xmin, ymin, xmax, ymax))
      buf = io.BytesIO()
      cropped_d.save(buf, format='JPEG')
      DESIGN = buf.getvalue()

      ymin, xmin, ymax, xmax = present_class["DESIGN_CENTRE"][1]
      ymin = ymin*frame_height
      xmin = xmin*frame_width
      ymax = ymax*frame_height
      xmax = xmax*frame_width
      cropped_d_c = img_grayscale.crop((xmin, ymin, xmax, ymax))
      buf = io.BytesIO()
      cropped_d_c.save(buf, format='JPEG')
      DESIGN_CENTRE = buf.getvalue()
    else:
      cv2.imwrite(PATH_TO_CROP + "/The_problematic_one's/Bracelate/{}.jpg".format(image_name), image)
    return DESIGN, DESIGN_CENTRE

def design_extraction_necklace(Tensors_tup, category_index, data):
  # Grab path to current working directory
  CWD_PATH = os.getcwd()

  # Output cropped
  CROP_FOLDER = 'Design'
  PATH_TO_CROP = os.path.join(CWD_PATH, CROP_FOLDER)
  if data.endswith('.db'):
    os.remove(data)
    print("removed")
  elif data.endswith(".jpeg") or data.endswith(".jpg") or data.endswith(".JPG") or data.endswith(".JPEG") \
      or data.endswith(".png") or data.endswith(".PNG") or data.endswith(".bmp") or data.endswith(".BMP"):
    # print("DATA________",data)

    image_str = data.replace('/','\\\\')
    image_str_list = image_str.split('\\\\')
    image_name = image_str_list[-1][:-4]

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(data)
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = Tensors_tup[0].run(
        [Tensors_tup[2], Tensors_tup[3], Tensors_tup[4], Tensors_tup[5]],
        feed_dict={Tensors_tup[1]: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    _ , _ , present_class = vis_util.visualize_boxes_and_labels_on_image_array(current_frame_number=1,
                                                                              image = image,
                                                                              mode=1,
                                                                              color_recognition_status=0,
                                                                              boxes=np.squeeze(boxes),
                                                                              classes=np.squeeze(classes).astype(np.int64),
                                                                              scores=np.squeeze(scores),
                                                                              category_index=category_index,
                                                                              use_normalized_coordinates=True,
                                                                              line_thickness=8,
                                                                              min_score_thresh=0.70)

    # Detected Objects in image---------------------------------------------------------
    # print(":::::----",present_class.keys())
    # print(":::::---- Imge:- ",image_name," detected label:- ",present_class.keys())

    # Crop the Detected Part from image-------------------------------------------------
    DESIGN = None
    if "DESIGN" in present_class.keys():
      img = Image.open(data)
      img_grayscale = ImageOps.grayscale(img)
      (frame_height, frame_width) = image.shape[:2]
      ymin, xmin, ymax, xmax = present_class["DESIGN"][1]
      ymin = ymin*frame_height
      xmin = xmin*frame_width
      ymax = ymax*frame_height
      xmax = xmax*frame_width
      cropped_d = img_grayscale.crop((xmin, ymin, xmax, ymax))
      buf = io.BytesIO()
      cropped_d.save(buf, format='JPEG')
      DESIGN = buf.getvalue()
    else:
      cv2.imwrite(PATH_TO_CROP + "/The_problematic_one's/Necklace/{}.jpg".format(image_name), image)
    return DESIGN

def design_extraction_all(Tensors_tup, category_index, data):
  # Grab path to current working directory
  CWD_PATH = os.getcwd()

  # Output cropped
  CROP_FOLDER = 'Design'
  PATH_TO_CROP = os.path.join(CWD_PATH, CROP_FOLDER)
  if data.endswith('.db'):
    os.remove(data)
    print("removed")
  elif data.endswith(".jpeg") or data.endswith(".jpg") or data.endswith(".JPG") or data.endswith(".JPEG") \
      or data.endswith(".png") or data.endswith(".PNG") or data.endswith(".bmp") or data.endswith(".BMP"):
    print("DATA________",data)

    image_str = data.replace('/','\\\\')
    image_str_list = image_str.split('\\\\')
    image_name = image_str_list[-1][:-4]

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(data)
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input

    (boxes, scores, classes, num) = Tensors_tup[0].run(
    [Tensors_tup[2], Tensors_tup[3], Tensors_tup[4], Tensors_tup[5]],
                feed_dict={Tensors_tup[1]: image_expanded})
    # Draw the results of the detection (aka 'visulaize the results')
    _ , _ , present_class = vis_util.visualize_boxes_and_labels_on_image_array(current_frame_number=1,
                                                                              image = image,
                                                                              mode=1,
                                                                              color_recognition_status=0,
                                                                              boxes=np.squeeze(boxes),
                                                                              classes=np.squeeze(classes).astype(np.int64),
                                                                              scores=np.squeeze(scores),
                                                                              category_index=category_index,
                                                                              use_normalized_coordinates=True,
                                                                              line_thickness=8,
                                                                              min_score_thresh=0.70)

    # Detected Objects in image---------------------------------------------------------
    # print(":::::----",present_class.keys())
    print(":::::---- Imge:- ",image_name," detected label:- ",present_class.keys())

    # Crop the Detected Part from image-------------------------------------------------
    DESIGN = None
    DESIGN_CENTRE = None
    if "DESIGN" in present_class.keys() and "DESIGN_CENTRE" in present_class.keys():
      img = Image.open(data)
      img_grayscale = ImageOps.grayscale(img)
      (frame_height, frame_width) = image.shape[:2]
      ymin, xmin, ymax, xmax = present_class["DESIGN"][1]
      ymin = ymin*frame_height
      xmin = xmin*frame_width
      ymax = ymax*frame_height
      xmax = xmax*frame_width
      cropped_d = img_grayscale.crop((xmin, ymin, xmax, ymax))
      buf = io.BytesIO()
      cropped_d.save(buf, format='JPEG')
      DESIGN = buf.getvalue()

      ymin, xmin, ymax, xmax = present_class["DESIGN_CENTRE"][1]
      ymin = ymin*frame_height
      xmin = xmin*frame_width
      ymax = ymax*frame_height
      xmax = xmax*frame_width
      cropped_d_c = img_grayscale.crop((xmin, ymin, xmax, ymax))
      buf = io.BytesIO()
      cropped_d_c.save(buf, format='JPEG')
      DESIGN_CENTRE = buf.getvalue()
    else:
      cv2.imwrite(PATH_TO_CROP + "/The_problematic_one's/Not_Catagorized/{}.jpg".format(image_name), image)
    return DESIGN, DESIGN_CENTRE