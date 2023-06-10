# Other Projects

---

### Face Emotion Recognition with Image Processing and Neural Network

*Convolutional Neural Network, Feature Extraction*

This project proposes a new **facial emotional recognition** model using a **convolutional neural network**. Therefore, a convolutional neural network based solution **combined with image processing** is used in classification of universal emotions.

Frontal face images are given as input to the system. To complete the training of the CNN network model, I used the **CK+** (Extended Cohn-Kanade dataset) CNN consists of four layers of convolution together with fully connected layers. The **features extracted** by the HOG(Histogram of Oriented Gradients), Convolutional Neural network (CNN) from facial expressions images were fused to develop the classification model through **training** by our proposed CNN model.

Additionally, for improved accuracy and prediction, a more advanced CNN network model was employed using the FER2013 dataset.

To explore project codes and documentation, feel free to click [here](https://github.com/mrtlckn/EmotionPrediction_Imageprocessing_NeuralNetwork).

---

### Empty Parking Space Detection

*Image Processing, OpenCV, pickle, numpy*

I developed an application using **image processing** techniques to detect empty parking space count and available parking spaces in a parking area. The project utilizes **OpenCV** library for video feed capture and processing.

User-selected parking space coordinates are stored as a **pickle** file. Image processing operations calculate vehicle count using **pixel counting**.

**Threshold** values determine presence of vehicles or empty spaces. Results are visualized with rectangles and text labels.

**Key technologies** used include Python, OpenCV, pickle, and numpy.

To explore project codes and documentation, feel free to click [here](https://github.com/mrtlckn/ParkingSpaceCounter).

---

### Instance Segmentation with Detectron 2

*LabelMe, Detectron2, Instance Segmentation*

The project utilizes **LabelMe**, a program for **annotating images**. It enables the labeling of images and generates separate JSON files for each image. A script is written to merge the **JSON files** from the train and test folders.

Following the steps outlined in the **Detectron2** documentation under "use **custom dataset**," the images and JSON files are introduced to Detectron2. The "get_train_cfg" function is used to specify the necessary information for the model, including the config file path, checkpoint URL, train and test dataset names, number of classes, device, and output directory.

To explore project codes and documentation, feel free to click [here](https://github.com/mrtlckn/detectron2).

---

### Object Detection with yoloV3 algorithm and Coco Dataset

The purpose of this project is object detection with CNN. For this reason I used yolov3 algorithm. YOLO uses features learned by a deep convolutional neural network to detect an object.

If the algorithm detects the object in the image, the detected objects are cropped. Then saved MySQL and pandas dataFrame some specific features which are "Tpye, Confidences, Coordinates, filepath, Okey_or_Not, time".

To explore project codes and documentation, feel free to click [here](https://github.com/mrtlckn/objectDetectionW_Yolov3).