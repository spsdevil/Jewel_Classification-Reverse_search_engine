# import the necessary packages
import numpy as np
import requests
import cv2
from PIL import Image, ImageDraw, ImageFont

# define the URL to our face detection API
url = "http://192.168.1.14:5000//Jewel_recommend"


# images by uploading an image directly
img = "Design/Rings/BDCK2153S.jpg"
image = cv2.imread(img)

# load our image and now use the face detection API to find faces in

# load image from System
payload = {"image": open(img, "rb")}
val = {'conf':70,'mode':6}
response = requests.post(url, files=payload, data=val).json()

# load image from server
# payload = {"url": "https://aon-face-api.s3-ap-southeast-1.amazonaws.com/combo3.jpg"}
# response = requests.post(url, data=payload).json()

print("Response: {}".format(response))