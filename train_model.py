import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

data = []
labels = []

dataset_path = "dataset/leapGestRecog"
print(os.listdir(dataset_path))
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    for gesture in os.listdir(person_path):
        gesture_path = os.path.join(person_path, gesture)

        for img in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img)

            image = cv2.imread(img_path)
            image = cv2.resize(image,(64,64))

            data.append(image)
            labels.append(gesture)

data = np.array(data)/255.0
labels = np.array(labels)

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)

X_train,X_test,y_train,y_test = train_test_split(
data,labels,test_size=0.2,random_state=42)

model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(
optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy']
)

model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test))

model.save("model.h5")

print("Model saved as model.h5")