import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import numpy as np

#Loading the saved model
cnn_cnd = tf.keras.models.load_model('cats_and_dogs.h5')

cnn_cnd.summary()

#preprocessing the Training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data_set = 'D:\p00hb\Documents\ADAMS Research\CNN_test\demo\dataset\\training_set/'

training_set = train_datagen.flow_from_directory(
    train_data_set,
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)
#preprocessing the Test set
test_data_set = 'D:\p00hb\Documents\ADAMS Research\CNN_test\demo\dataset\\test_set/'
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    test_data_set,
    target_size= (64,64),
    batch_size= 32,
    class_mode= 'binary'
)

'''
Part 4 - Making a single prediction
'''
test_image = image.load_img('D:\p00hb\Documents\ADAMS Research\CNN_test\demo\dataset\single_prediction\\cat_or_dog_1.jpg/', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn_cnd.predict(test_image/255.0)
training_set.class_indices
if result[0][0] > 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)

#Evaluating the model
loss, acc = cnn_cnd.evaluate(x = test_set, y = None, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))