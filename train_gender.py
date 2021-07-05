import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, LayerNormalization
import pandas
import glob
import matplotlib.pyplot as plt
import os

image_size = (64, 64)
batch_size = 32
epochs = 15
model_name = "gender_model"

folders = ["UTKFace/"]
countCat = {0:0, 1:0}
class_weight = {0:1, 1:1}
data, labels = [], []
for folder in folders:
    for file in glob.glob(folder+"*.jpg"):
        file = file.replace(folder, "")
        age, gender = file.split("_")[0:2]
        age, gender = int(age), int(gender)
        countCat[gender]+=1
        filepath = folder + file
        label = str(gender)
        data.append(filepath)
        labels.append(label)
        print(filepath, label)

minVal = min(countCat.values())
for key in class_weight:
    class_weight[key]/=countCat[key]
    class_weight[key]*=minVal
print(class_weight)
train_df = pandas.DataFrame(data={"filename": data, "class": labels})

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filename",
        y_col="class",
        target_size=image_size,
        batch_size=batch_size,
        subset='training',
        class_mode='categorical')

validation_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filename",
        y_col="class",
        target_size=image_size,
        batch_size=batch_size,
        subset='validation',
        class_mode='categorical')

print(train_generator.class_indices)
if os.path.exists(model_name):
    print("Load: " + model_name)
    classifier = load_model(model_name)
else:
    #ref to https://techvidvan.com/tutorials/gender-age-detection-ml-keras-opencv-cnn/
    classifier = Sequential()
    classifier.add(Conv2D(input_shape=image_size + (3,), filters=96, kernel_size=(7, 7), strides=4, padding='valid', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    classifier.add(LayerNormalization())
    classifier.add(Conv2D(filters=256, kernel_size=(5, 5), strides=1, padding='same', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    classifier.add(LayerNormalization())
    classifier.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    classifier.add(LayerNormalization())

    classifier.add(Flatten())
    classifier.add(Dense(units=512, activation='relu'))
    classifier.add(Dropout(rate=0.25))
    classifier.add(Dense(units=512, activation='relu'))
    classifier.add(Dropout(rate=0.25))
    classifier.add(Dense(units=2, activation='softmax'))

    #classifier = tf.keras.applications.MobileNetV3Small(include_top=True, weights=None, input_tensor=None, input_shape=image_size + (3,), pooling='max', classes=2)
    classifier.summary()
    classifier.compile(loss='categorical_crossentropy', metrics=['accuracy'])
history = classifier.fit(train_generator, steps_per_epoch=train_generator.samples//batch_size, epochs=epochs, validation_data=validation_generator, validation_steps=validation_generator.samples//batch_size, class_weight=class_weight)
classifier.save(model_name)

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['accuracy'])
plt.legend(['loss', 'acc'])
plt.savefig("learning-gender.png")
plt.show()
plt.close()
