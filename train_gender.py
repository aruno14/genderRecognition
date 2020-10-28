import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas
import glob

image_size = (48, 48)
batch_size = 32
epochs = 15

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
        data.append(folder + file)
        labels.append(str(gender))

minVal = min(countCat.values())
for key in class_weight:
    class_weight[key]/=countCat[key]
    class_weight[key]*=minVal
print(class_weight)
train_df = pandas.DataFrame(data={"filename": data, "class": labels})

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filename",
        y_col="class",
        target_size=image_size,
        batch_size=batch_size,
        subset='training',
        class_mode='categorical')

validation_generator = test_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filename",
        y_col="class",
        target_size=image_size,
        batch_size=batch_size,
        subset='validation',
        class_mode='categorical')

print(train_generator.class_indices)
classifier = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=True, weights=None, input_tensor=None, input_shape=image_size + (3,), pooling=None, classes=2)
classifier.compile(loss='categorical_crossentropy', metrics=['accuracy'])
classifier.fit(train_generator, steps_per_epoch=train_generator.samples//batch_size, epochs=epochs, validation_data=validation_generator, validation_steps=validation_generator.samples//batch_size, class_weight=class_weight)
classifier.save("gender_model")
