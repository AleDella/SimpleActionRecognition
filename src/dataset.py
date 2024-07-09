import cv2
import math
import os
import numpy as np
from .methods import PersonDetector



def get_frames(video_path, sequence_length=20, image_width=64, image_height=64, normalize=False):
    '''
    Function that extracts the frames from the provided videos
    
    Args:
        video_path (str): path to the video
        sequence_length (int): length of the sequences of frames
        image_width (int): width in pixels
        image_height (int): height in pixels
        normalize (bool): image normalization
    Returns:
        frames (list[np.array]): list of frames in the video
    '''
    
    frames = []
    cap = cv2.VideoCapture(video_path)
    # Check if caption opened successfully 
    if (cap.isOpened()== False): 
        print(f"Error opening {video_path}!")
    
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # Compute the interval for the division of the videos to extract only n frames from the video
    frame_window = max(math.floor(total_frames/sequence_length),1)
    
    for f_idx in range(sequence_length):
        # Set the position of the frame we want
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx*frame_window)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        success, frame = cap.read()
        if not success:
            break
        # Frame pre-processing (resizing, normalization)
        resized_frame = cv2.resize(frame, (image_height, image_width))
        
        if normalize:
            resized_frame = resized_frame/255
        
        frames.append(resized_frame)
    
    cap.release()
    
    return frames


def get_cropped_frames(video_path, sequence_length=20, image_width=64, image_height=64, normalize=False):
    '''
    Function that extracts the frames from the provided videos
    which are already cropped and saved a jpg
    
    Args:
        video_path (str): path to the video
        sequence_length (int): length of the sequences of frames
        image_width (int): width in pixels
        image_height (int): height in pixels
        normalize (bool): image normalization
    Returns:
        frames (list[np.array]): list of frames in the video
    '''
    
    frames = []
    total_frames = os.listdir(video_path)
    if total_frames != []:
        for f_idx in range(sequence_length):
            # Set the position of the frame we want
            frame_name = total_frames[f_idx]
            frame = cv2.imread(os.path.join(video_path,frame_name))
            # Frame pre-processing (resizing, normalization)
            resized_frame = cv2.resize(frame, (image_height, image_width))
            
            if normalize:
                resized_frame = resized_frame/255

            frames.append(resized_frame)
    
    return frames

def raw_dataset_extraction(dataset_dir, n_classes, sequence_length, image_width, image_height, cropped=False):
    '''
    Function that creates the dataset starting from the data of the toy dataset
    
    Args:
        dataset_dir (str): path to the dataset
        n_classes (int): number of classes to extract from the dataset
        sequence_length (int): length of the sequences of frames
        image_width (int): width in pixels
        image_height (int): height in pixels
        cropped (bool): show if the dataset is the cropped one or not
    Returns:
        frame_lists (np.array): list of collection of frames
        labels (np.array): list of labels for the videos
        paths (list): list of filenames of the videos
    '''
    frame_lists = []
    labels = []
    paths = []
    classes = os.listdir(dataset_dir)
    for cl_idx, cl in enumerate(classes):
        if cl_idx<n_classes:
            cl_videos = os.listdir(os.path.join(dataset_dir,cl))
            for filename in cl_videos:
                fullpath = os.path.join(dataset_dir,cl,filename)
                if cropped:
                    frames = get_cropped_frames(fullpath, sequence_length, image_width, image_height)
                else:
                    frames = get_frames(fullpath, sequence_length, image_width, image_height)
                if len(frames) == sequence_length:
                    frame_lists.append(frames)
                    labels.append(cl_idx)
                    paths.append(fullpath)
                
    frame_lists = np.array(frame_lists)
    labels = np.array(labels)
    
    return frame_lists, labels, paths


def crop_dataset(dataset_dir, new_dataset_dir, n_classes):
    '''
    Function that crops the dataset starting from the data of the toy dataset
    and saves the cropped images in folders
    
    Args:
        dataset_dir (str): path to the dataset
        new_dataset_dir (str): path to the new dataset
        n_classes (int): number of classes to extract from the dataset
    Returns:
        None
    '''
    model = PersonDetector(image_size=[300,300])
    classes = os.listdir(dataset_dir)
    for cl_idx, cl in enumerate(classes):
        if cl_idx<n_classes:
            # Create new directories for the dataset
            new_folder_path = os.path.join(new_dataset_dir,cl)
            if not os.path.exists(new_folder_path):
                os.mkdir(new_folder_path)
            if cl_idx == 0:
                cl_videos = os.listdir(os.path.join(dataset_dir,cl))[10:]
            else:
                cl_videos = os.listdir(os.path.join(dataset_dir,cl))
            for filename in cl_videos:
                fullpath = os.path.join(dataset_dir,cl,filename)
                new_video_path = os.path.join(new_folder_path, filename)
                if not os.path.exists(new_video_path):
                    os.mkdir(new_video_path)
                cap = cv2.VideoCapture(fullpath)
                # Check if caption opened successfully 
                if (cap.isOpened()== False): 
                    print(f"Error opening {fullpath}!")
                
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                total_frames = 100
                for f_idx in range(int(total_frames)):
                    # Set the position of the frame we want
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    success, frame = cap.read()
                    if not success:
                        break
                    cropped_frame = model.detect_from_frame(frame)
                    frame_name = os.path.join(new_video_path,str(f_idx)+'.jpg')
                    try:
                        cv2.imwrite(frame_name, cropped_frame[0])
                    except:
                        continue

                cap.release()




def convert_to_frames(frames, labels, grayscale=True):
    '''
    Function that converts the dataset from a list of videos to a list of frames
    '''
    new_frames = []
    new_labels = []
    for vd,lb in zip(frames,labels):
        for fr in vd:
            if grayscale:
                new_frames.append(cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY))
            else:
                new_frames.append(fr)
            new_labels.append(lb)
    
    new_frames = np.array(new_frames)
    new_labels = np.array(new_labels)
    
    return new_frames, new_labels