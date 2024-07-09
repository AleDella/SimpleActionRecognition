'''
File used to evaluate different algorithms once they're saved
'''
# Imports
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Custom libraries 
from src import *


if __name__ == '__main__':
    # Variables for execution
    raw_dataset_dir = 'dataset/ToyDataset' # Parent directory of the dataset
    cropped_dataset_dir = 'dataset/cropped_dataset'
    n_classes = 2 # Number of classes to consider (influences also the video retrieved)
    sequence_length = 16 # Number of frames taken for each video
    image_width = 200 # Pixels
    image_height = 200 # Pixels
    seed = 2024 
    test_size = 0.2 # Percentage of the total number of videos

    # Flags for the methods to analyze
    create_dataset = False
    obj_det_flag = False
    grayscale = False
    train = False
    bow_flag = False
    lrcn_flag = True

    # Setting seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Prune the classes for plotting based on n_classes
    classes = ['Backhand', 'Forehand']
    classes = classes[:n_classes]

    if create_dataset:
        crop_dataset(raw_dataset_dir, 'dataset\\cropped_dataset', 2)
    else:
        if obj_det_flag:
            dataset_dir = cropped_dataset_dir
        else:
            dataset_dir = raw_dataset_dir
        # Extract the frames we want with the resolution wanted
        frame_lists, labels, _ = raw_dataset_extraction(dataset_dir, n_classes, sequence_length, image_width, image_height, obj_det_flag)
        # Split test and train (Take only test for evaluation)
        frame_train, frame_test, label_train, label_test = train_test_split(frame_lists, labels, test_size=test_size, shuffle=True, random_state=seed)

        if bow_flag:
            # Use bag of visual words approach
            print("Using bag of visual words pipeline...")
            b = BoW(classes=classes)
            if train:
                # Convert the frame list into frame-wise dataset instead of video-wise dataset
                frame_train, label_train = convert_to_frames(frame_train, label_train, grayscale=grayscale)
                features,descriptors = b.train(frame_train,label_train)
                print("Training done!")
                frame_test, label_test = convert_to_frames(frame_test, label_test)
                acc = b.test(frame_test, label_test)
                if obj_det_flag:
                    name = 'trained_models/BoW_model_obj_det'
                    b.save(dirname=name, features=features, descriptors=descriptors)
                else:
                    b.save(features=features, descriptors=descriptors)
                print("Model saved!")
            else:
                # Do the inference on the test set
                if obj_det_flag:
                    name = 'trained_models/BoW_model_obj_det_128_128'
                    b.load(dirname=name)
                else:
                    b.load()
                frame_test, label_test = convert_to_frames(frame_test, label_test)
                acc = b.test(frame_test, label_test)
        if lrcn_flag:
            # Use LRCN deep learning approach
            print("Evaluating ResNet-CRNN approach...")
            if train:
                run_training_ResNetCRNN('dataset\\ToyDataset', classes, is_cropped=obj_det_flag, epochs=40, dropout_p=0.0)
            else:
                cnn_encoder, rnn_decoder = load_model('trained_models\\ResNetCRNN', classes)
                all_y, all_y_pred = run_testing_ResNetCRNN(cnn_encoder, rnn_decoder, 'dataset\\ToyDataset', classes)
                plotConfusionMatrix(all_y.cpu(), all_y_pred.cpu(), classes, True, 'LRCN Confusion Matrix')
                plt.show()