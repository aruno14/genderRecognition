import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy

image_size = (64, 64)
model_name = "gender_model_gray"

model = load_model(model_name)
for imageName in ["test.jpg"]:
    print(imageName)
    test_image = image.load_img(imageName, target_size=image_size, color_mode='grayscale')
    test_image = image.img_to_array(test_image)
    test_image = test_image.astype('float32')
    test_image /= 255.0
    test_image = numpy.expand_dims(test_image, axis=0)
    prediction = model.predict(test_image)
    print(prediction)
