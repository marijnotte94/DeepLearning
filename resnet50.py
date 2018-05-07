import pandas as pd
import numpy as np
import os
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras import optimizers
from keras.applications import resnet50


def pretrained_model(img_shape, num_classes):
    resnet_model = resnet50.ResNet50(include_top=False, weights='imagenet')

    # Create your own input format
    keras_input = Input(shape=img_shape, name='image_input')

    # Use the generated model
    output = resnet_model(keras_input)

    # Add the fully-connected layers
    x = Flatten(name='flatten')(output)
    #x = Dense(num_classes, activation='softmax', name='predictions')(x)
    x = Dense(1, name='predictions')(x)

    # Create your own model
    pretrained_model = Model(inputs=keras_input, outputs=x)
    pretrained_model.compile(loss='mean_squared_error', optimizer=optimizers.adam(lr=0.001), metrics=['accuracy'])

    return pretrained_model

def load_data(filenames):
    x_train = []
    for file in filenames:
        img = image.load_img(train_dir + '/' + file, target_size=(224, 224))
        array_img = image.img_to_array(img)
        array_img = np.expand_dims(array_img, axis=0)
        x_train.append(array_img)

    x_train = np.vstack(x_train)
    x_train = preprocess_input(x_train)

    return x_train

data_dir = 'data/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# read csv with all data info
filename = 'data/all_data_info.csv'
df = pd.read_csv(filename)

# drop rows that have no exact date
df = df[pd.to_numeric(df['date'], errors='coerce').notnull()]
df['date'] = pd.to_numeric(df['date'], errors='coerce', downcast='signed')

# drop rows that are in test set
df = df[df['in_train']]

filenames = list(df['new_filename'].values)
labels = list(df['date'].values / 100)
print(labels)

del df

num_samples = 10000
x_train = load_data(filenames[1:num_samples])
y_train = np.asarray(labels[1:num_samples])

# data augmentation
# datagen = image.ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)
#
# datagen.fit(x_train)

pretrained_model = pretrained_model(x_train.shape[1:], len(set(y_train)))

# training the model
hist = pretrained_model.fit(x_train, y_train, batch_size=8, epochs=1, validation_split=0.1, verbose=1)
#hist = pretrained_model.fit_generator(datagen.flow(x_train, y_train, batch_size=8), epochs=1)

img = image.load_img(test_dir + '/' + '0.jpg', target_size=(224, 224))
array_img = image.img_to_array(img)
array_img = np.expand_dims(array_img, axis=0)
prediction = pretrained_model.predict(array_img)
print(str(prediction))
