import numpy as np
from src import raw_dataset_extraction, plotConfusionMatrix
from tf_keras.layers import TimeDistributed, Conv2D, Dense, MaxPooling2D, Flatten, LSTM, Dropout, BatchNormalization
from tf_keras import models
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tf_keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    dataset_dir = 'dataset\\ToyDataset'
    n_classes = 2
    sequence_length = 20
    image_width = 128
    image_height = 128
    train=False
    
    frames, labels, _ = raw_dataset_extraction(dataset_dir, n_classes, sequence_length, image_width, image_height, cropped=False, grayscale=True)
    coded_labels = to_categorical(labels)
    frame_train, frame_test, label_train, label_test = train_test_split(frames, coded_labels, test_size=0.2, shuffle=True, random_state=2024)

    

    model_cnlst = models.Sequential()
    model_cnlst.add(TimeDistributed(Conv2D(128, (3, 3), strides=(1,1),activation='relu'),input_shape=(sequence_length, image_width, image_height, 1)))
    model_cnlst.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1,1),activation='relu')))
    model_cnlst.add(TimeDistributed(MaxPooling2D(2,2)))
    model_cnlst.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1,1),activation='relu')))
    model_cnlst.add(TimeDistributed(Conv2D(32, (3, 3), strides=(1,1),activation='relu')))
    model_cnlst.add(TimeDistributed(MaxPooling2D(2,2)))
    model_cnlst.add(TimeDistributed(BatchNormalization()))


    model_cnlst.add(TimeDistributed(Flatten()))
    model_cnlst.add(Dropout(0.2))

    model_cnlst.add(LSTM(64,return_sequences=False,dropout=0.2)) # used 32 units

    model_cnlst.add(Dense(64,activation='relu'))
    model_cnlst.add(Dense(32,activation='relu'))
    model_cnlst.add(Dropout(0.2))
    model_cnlst.add(Dense(2, activation='softmax'))
    model_cnlst.summary()

    if train:
        callbacks_list_cnlst=[EarlyStopping(monitor='acc',patience=3),
                    ModelCheckpoint(
                    filepath='cnn_lstm_model_new3.h5',
                    monitor='val_loss',
                    save_best_only=True),
                        ReduceLROnPlateau(monitor = "val_loss", factor = 0.1, patience = 8)
                    ]


        model_cnlst.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['acc'])
        # Training:
        history_new_cnlst=model_cnlst.fit(frame_train,label_train,batch_size=10,epochs=20,
                                validation_data=(frame_test,label_test),
                            callbacks=callbacks_list_cnlst)
        model_cnlst.save_weights('cnn_lstm_model_new3.h5')
    else:
        model_cnlst.load_weights('cnn_lstm_model_new3.h5')
        print("Model loaded!")
        model_cnlst.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['acc'])
        res,res1 = model_cnlst.evaluate(frame_test, label_test)
        print(frame_test.shape[0])
        prediction = model_cnlst.predict(frame_test)
        print(np.argmax(prediction, axis=1))
        plotConfusionMatrix(np.argmax(label_test, axis=1), np.argmax(prediction, axis=1), classes=['Backhand', 'Forehand'], normalize=True, title="LRCN Confusion Matrix")
        plt.show()
        