'''
File used to evaluate different algorithms once they're saved
'''
# Imports
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
# Custom libraries 
from src import *


if __name__ == '__main__':
    f = open('params.json','r')
    parameters = json.load(f)

    # Setting seed for reproducibility
    seed = 2024 
    np.random.seed(seed)
    random.seed(seed)
    n_classes = len(parameters['dataset']['classes'])
    

    if bool(parameters['experiment']['create_dataset_flag']):
        crop_dataset(parameters['dataset']['raw_dataset_dir'], parameters['dataset']['cropped_dataset_dir'], 2)
    else:
        if bool(parameters['experiment']['object_detection_flag']):
            dataset_dir = parameters['dataset']['cropped_dataset_dir']
        else:
            dataset_dir = parameters['dataset']['raw_dataset_dir']
        # Extract the frames we want with the resolution wanted
        frame_lists, labels, _ = raw_dataset_extraction(dataset_dir, n_classes, parameters['dataset']['video_frames'], parameters['dataset']['image_width'], parameters['dataset']['image_height'], bool(parameters['experiment']['object_detection_flag']))
        # Split test and train (Take only test for evaluation)
        frame_train, frame_test, label_train, label_test = train_test_split(frame_lists, labels, test_size=parameters['experiment']['test_dataset_size'], shuffle=True, random_state=seed)

        if bool(parameters['experiment']['bag_of_words_flag']):
            # Use bag of visual words approach
            print("Using bag of visual words pipeline...")
            b = BoW(classes=parameters['dataset']['classes'])
            if bool(parameters['experiment']['train']):
                # Convert the frame list into frame-wise dataset instead of video-wise dataset
                frame_train, label_train = convert_to_frames(frame_train, label_train, grayscale=bool(parameters['dataset']['grayscale_images']))
                features,descriptors = b.train(frame_train,label_train)
                print("Training done!")
                frame_test, label_test = convert_to_frames(frame_test, label_test)
                acc = b.test(frame_test, label_test)
                b.save(dirname=parameters['bow']['savename'], features=features, descriptors=descriptors)
                print("Model saved!")
            else:
                # Do the inference on the test set
                b.load(dirname=parameters['bow']['savename'])
                frame_test, label_test = convert_to_frames(frame_test, label_test)
                acc = b.test(frame_test, label_test)
        if bool(parameters['experiment']['lrcn_flag']):
            # Use LRCN deep learning approach
            print("Evaluating ResNet-CRNN approach...")
            if bool(parameters['experiment']['train']):
                run_training_ResNetCRNN(parameters['dataset']['raw_dataset_dir'], parameters['dataset']['classes'], is_cropped=parameters['experiment']['object_detection_flag'], epochs=parameters['experiment']['epochs'], dropout_p=parameters['experiment']['dropout'])
            else:
                cnn_encoder, rnn_decoder = load_model(parameters['lrcn']['savename'], parameters['dataset']['classes'])
                all_y, all_y_pred = run_testing_ResNetCRNN(cnn_encoder, rnn_decoder, parameters['dataset']['raw_dataset_dir'], parameters['dataset']['classes'])
                plotConfusionMatrix(all_y.cpu(), all_y_pred.cpu(), parameters['dataset']['classes'], True, 'LRCN Confusion Matrix')
                plt.show()