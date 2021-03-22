import os
import cv2
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from PIL import Image

# Design extraction of images present in a dir ------------------------------------
def classification(detection_graph, label_lines, dir_path):
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

      for data in tqdm(os.listdir(dir_path)):
        if data.endswith(".jpeg") or data.endswith(".jpg") or data.endswith(".JPG") or data.endswith(".JPEG") \
              or data.endswith(".png") or data.endswith(".PNG") or data.endswith(".bmp") or data.endswith(".BMP"):
          img_ = os.path.join(dir_path, data)
          # print(img_)
          image_data = tf.gfile.FastGFile(img_, 'rb').read()
          
          predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
          top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
          
          # output -----
          for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            if score>0.7:
              # print('%s (score = %.5f)' % (human_string, score))
              img_pil = Image.open(img_)
              output_path = "Design/" + human_string +'/' + img_.split('\\')[-1]
              img_pil.save(output_path)
  return "True"
