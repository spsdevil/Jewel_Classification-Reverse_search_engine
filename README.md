# Jewel_Classification_Reverse_search_engine
This is complete project to classify the jewelries into different categories like Rings, Necklace, Pendant, etc., using CNN model for image classification and then using image similarity and vector comparison metrics reverse image search engine for jewellery.

Vgg-16 model is used as base model to train classification model on the custom dataset of the jewellery with 5 categories named as Ring, Bracelet, Pendant, Necklace, and Earring.
Same classification model trained on custom dataset is used for the features extraction of the images, before feature extraction 5 different object detection model is used over different categories to exactly identify the boundries of the jewellery in the image, and then cropped image is sent to the classification model to extract features, these embedded vectors are extracted by removing last layer that is softmax(classifier) to obtain scalable features of size 2048, these embedded vectors are then stored in the pickle file.

once we hace vector file for all the image accross all the categories. Note:-> there are total 5 vectors files for 5 categories, one for each, along with 5 neighbour files which store the location of the image on same index of vector file, we use these file to map those images which are similar to our query image.

Same model is used on each query image to extract features and then there is similarity check for each vectors stored in the vector file using Cosine_distance.

top_k images which matches to the query images are returned at the final step.

all the APIs are available here including:- Image Vectorizer, Query_Search_API, Classification, and few more internal APIs to manipulate the result provided the model as feedback.  
