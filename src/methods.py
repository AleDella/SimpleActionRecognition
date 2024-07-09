'''
File that implements the classes for all the methods of this project
'''
import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from pickle import dump,load
from skl2onnx import to_onnx
from .metrics import plotConfusionMatrix, findAccuracy

class PersonDetector:
    '''
    Class that implements the detector for yolo
    '''
    def __init__(self,image_size=None,stream_mode=True,show=False):
        self.model = YOLO('yolov9e-seg.pt')
        self.image_size = image_size
        self.stream_mode = stream_mode
        self.show = show
    
    def detect_from_frame(self, frame, max_targets=1):
        '''
        Detection from a single frame

        Args:
            frame (numpy.array): frame for the detection
            max_targets (int): maximum targetss detected
        Returns:
            cropped_frames: list of images containing the targets
        '''
        if self.image_size is not None:
            results = self.model(frame, imgsz=self.image_size, classes=[0], show=self.show, stream=self.stream_mode)
        else:
            results = self.model(frame, classes=[0], show=self.show, stream=self.stream_mode)
        
        cropped_frames = []

        for r in results:
            boxes = r.boxes
            for i,box in enumerate(boxes):
                if i<max_targets:
                    x1,y1,x2,y2 = box.xyxy[0]
                    tmp = frame[int(y1):int(y2), int(x1):int(x2)]
                    cropped_frames.append(tmp)

        cropped_frames = np.array(cropped_frames)

        return cropped_frames
    
    def detect_from_video(self, video_path, max_targets=1, crop_size=128):
        '''
        Detection from a single video

        Args:
            video_path (str): video path for the detection
            max_targets (int): maximum targetss detected
        Returns:
            cropped_video: list of images containing the targets
        '''
        cap = cv2.VideoCapture(video_path)
        cropped_video = []
        # Check if caption opened successfully 
        if (cap.isOpened()== False): 
            print(f"Error opening {video_path}!")
        
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = 10
        for f_idx in range(int(total_frames)):
            # Set the position of the frame we want
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)    
            success,frame = cap.read()
            if success:
                cropped_frames = self.detect_from_frame(frame,max_targets)
                cropped_video.append(cv2.resize(cropped_frames[0], (crop_size, crop_size)))
        cap.release()
        cropped_video = np.array(cropped_video)

        return cropped_video
    

class BoW:
    '''
    Class for the bag of visual words approach
    '''
    def __init__(self, descriptors='orb', n_clusters=50, kernel_type='linear', classes=['Backhand', 'Forehand', 'Serve']):
        match descriptors:
            case 'orb':
                self.descriptors = cv2.ORB_create()
                self.descriptor_size = 32
            case 'sift':
                self.descriptors = cv2.SIFT_create()
                self.descriptor_size = 128
            case _:
                print("WARNING: Descriptor not implemented yet! Functions will not work!")
        self.n_clusters = n_clusters
        self.kernel_type = kernel_type
        self.classes = classes
        self.model = None
        self.scale = None
        self.kmeans = None
    
    def train(self, x, y):
        '''
        Method to train the model

        Args:
            x: inputs
            y: labels
        '''
        descriptor_list = []
        train_labels = y
        image_count = len(y)
        
        for img in x:
            des = self._getDescriptors(img)
            descriptor_list.append(des)

        # Stack the descriptors
        descriptors = self._vstackDescriptors(descriptor_list)
        # Clustering
        self.kmeans = self._clusterDescriptors(descriptors)
        # Extract features
        features = self._extractFeatures(self.kmeans, descriptor_list, image_count)
        # Normalize
        self.scale = StandardScaler().fit(features)
        features = self.scale.transform(features)
        # Train the SVM model
        self.model = self._findSVM(features, train_labels)

        return features,descriptors

    def test(self, x, y):
        '''
        Function for testing the model
        '''
        if (self.model is None) or (self.scale is None) or (self.kmeans is None):
            print("The model need to be trained or loaded first!")
            exit(0)
        
        count = 0
        true = []
        descriptor_list = []

        name_dict =	{str(i):cl for i,cl in enumerate(self.classes)}
        start_time = time.time()
        for lab,img in zip(y, x):
            des = self._getDescriptors(img)

            if(des is not None):
                count += 1
                descriptor_list.append(des)
                true.append(name_dict[str(lab)])

        
        test_features = self._extractFeatures(self.kmeans, descriptor_list, count)

        test_features = self.scale.transform(test_features)
        
        kernel_test = test_features
        
        predictions = [name_dict[str(int(i))] for i in self.model.predict(kernel_test)]
        print(f'Total time for inference on {len(y)} set: {time.time()-start_time}')
        plotConfusionMatrix(true, predictions, classes=self.classes, title='BoW Confusion matrix', normalize=True)
        plt.show()
        return findAccuracy(true, predictions)

    def save(self, dirname='trained_models/BoW_model', features=None, descriptors=None, is_onnx=False):
        '''
        Method to save the BoW model
        '''
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        if is_onnx:
            kmeans = to_onnx(self.kmeans, descriptors[:1].astype(np.int64))
            scale = to_onnx(self.scale, features[:1]) # need to be modified
            svm = to_onnx(self.model, features[:1])   # need to be modified

            with open(os.path.join(dirname,"kmeans.onnx"), "wb") as f:
                f.write(kmeans.SerializeToString())
            with open(os.path.join(dirname,"scale.onnx"), "wb") as f:
                f.write(scale.SerializeToString())
            with open(os.path.join(dirname,"svm.onnx"), "wb") as f:
                f.write(svm.SerializeToString())
        else:
            with open(os.path.join(dirname,"kmeans.pkl"), "wb") as f:
                dump(self.kmeans, f, protocol=5)
            with open(os.path.join(dirname,"scale.pkl"), "wb") as f:
                dump(self.scale, f, protocol=5)
            with open(os.path.join(dirname,"svm.pkl"), "wb") as f:
                dump(self.model, f, protocol=5)
        
    def load(self, dirname='trained_models/BoW_model', is_onnx=False):
        '''
        Method to load the BoW model
        '''
        if is_onnx:
            print("ONNX loading not implemented yet")
        else:
            with open(os.path.join(dirname,"kmeans.pkl"), "rb") as f:
                self.kmeans = load(f)
            with open(os.path.join(dirname,"scale.pkl"), "rb") as f:
                self.scale = load(f)
            with open(os.path.join(dirname,"svm.pkl"), "rb") as f:
                self.model = load(f)
        
    # SUPPORT METHODS
    def _getDescriptors(self,img):
        _, des = self.descriptors.detectAndCompute(img, None)
        return des
    def _vstackDescriptors(self,descriptor_list):
        descriptors = np.array(descriptor_list[0])
        for descriptor in descriptor_list[1:]:
            if descriptor is not None:
                descriptors = np.vstack((descriptors, descriptor))
            else:
                print("WARNING: Image to small to retrieve a descriptor!")

        return descriptors
    
    def _clusterDescriptors(self,descriptors):
        kmeans = KMeans(n_clusters=self.n_clusters).fit(descriptors)
        return kmeans
    
    def _extractFeatures(self, kmeans, descriptor_list, image_count):
        im_features = np.array([np.zeros(self.n_clusters) for i in range(image_count)])
        for i in range(image_count):
            if descriptor_list[i] is not None:
                for j in range(len(descriptor_list[i])):
                    feature = descriptor_list[i][j]
                    feature = feature.reshape(1, self.descriptor_size)
                    idx = kmeans.predict(feature)
                    im_features[i][idx] += 1

        return im_features
    
    def _findSVM(self, im_features, train_labels):
        features = im_features
        if(self.kernel_type == "precomputed"):
            features = np.dot(im_features, im_features.T)
        
        params = self._svcParamSelection(features, train_labels, 5)
        C_param, gamma_param = params.get("C"), params.get("gamma")
        print(C_param, gamma_param)
        class_weight = {
            0: (807 / (7 * 140)),
            1: (807 / (7 * 140)),
            2: (807 / (7 * 133)),
            3: (807 / (7 * 70)),
            4: (807 / (7 * 42)),
            5: (807 / (7 * 140)),
            6: (807 / (7 * 142)) 
        }
    
        svm = SVC(kernel = self.kernel_type, C =  C_param, gamma = gamma_param, class_weight = class_weight)
        svm.fit(features, train_labels)
        return svm
    
    def _svcParamSelection(self, X, y, nfolds):
        Cs = [0.5, 0.1, 0.15, 0.2, 0.3]
        gammas = [0.1, 0.11, 0.095, 0.105]
        param_grid = {'C': Cs, 'gamma' : gammas}
        grid_search = GridSearchCV(SVC(kernel=self.kernel_type), param_grid, cv=nfolds)
        grid_search.fit(X, y)
        grid_search.best_params_
        return grid_search.best_params_