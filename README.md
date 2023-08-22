# Unconstrained-Face-Recognition
This was my final year undergraduate project from Feb 2021 to July 2021. I also extended this project to a research project till May 2022.

To download the datasets for this project, go to https://drive.google.com/drive/folders/1X8ja6E3wW-43KR7n21ZVP5TrmjCLOjve?usp=drive_link

To download the pre-trained models, go to https://drive.google.com/drive/folders/1ZY6aDORBWMnszvORBtV6ehFaxjPNE6O7?usp=sharing, https://drive.google.com/drive/folders/1SM0uGsAN4ZKbNjA8DBI5eO7Esfx1aABZ?usp=sharing

This project is run in Python 3.5 version. This version is compatible with all the libraries in the requirements.txt file. Download all the files and store them inside another folder. You can name it 'Face_ID'.

The steps to run are given below:

1. Python 3.5 is to be installed before-hand


$ PATH_TO_PYTHON=/usr/bin/python3.5

2. Virtual environment is created

$ virtualenv -p $PATH_TO_PYTHON Face_ID

3. $ cd Face_ID
   
   $ source bin/activate

5. Install all the necessary libraries (TensorFlow 1.7 is required for this project)

$ pip install -r requirements.txt

FACE IDENTIFICATION

1. python 'path to the code file' 'path to the original dataset folder' 'path to the aligned dataset folder'

For example the script to be run is align_dataset_mtcnn whose path is 'facenet/src/align_dataset_mtcnn' and dataset 'raw' has path facenet/dataset/raw and 'aligned' folder where images will be aligned and stored has path facenet/dataset/aligned

$ python facenet/src/align_dataset_mtcnn.py facenet/dataset/raw facenet/dataset/aligned --image_size 160 --margin 32

every image is resized to 160 x 160 with margin 32

2. Now we train a classifier 

$ python facenet/src/classifier.py TRAIN facenet/dataset/aligned facenet/src/20180402-114759/facenet/src/20180402-114759/my_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 10 --nrof_train_images_per_class 10 --use_split_dataset

3. Its now time to test 
For example, to test image 'test1.jpg'

$ python facenet/src/face_recognition_image.py facenet/dataset/test-images/test1.jpg

Link for reference to perform Face identification

https://appliedmachinelearning.blog/2018/10/30/yet-another-face-recognition-demonstration-on-images-videos-using-python-and-tensorflow/


TESTING ACCURACY AND VALIDATION RATE

1. Arrange the images with clear labels same as format of lfw dataset

For example, For Caltech face dataset we rearrange the dataset same as lfw format

$ python facenet/src/organ_data.py --data_dir facenet/dataset/Caltech_faces/ --save_dir facenet/dataset/Caltech_faces1/

2. Align the images(use the directory Caltech_faces1) and save it in a different directory (Create the new directory before aligning)

$ python facenet/src/align_dataset_mtcnn.py facenet/dataset/Caltech_faces1 facenet/dataset/lfw_aligned --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25


3. Run accuracy test on desired model (in this case model 20180408-102900)

$ python facenet/src/validate_on_caltech.py facenet/dataset/caltech_aligned facenet/src/20180408-102900 --distance_metric 1 --use_flipped_images --subtract_mean --use_fixed_image_standardization

