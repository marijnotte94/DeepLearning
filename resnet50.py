import numpy as np
from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.applications import resnet50

def pretrained_model(img_shape, num_classes):
    resnet_model = resnet50.ResNet50(include_top=False, weights='imagenet')

    # Create your own input format
    keras_input = Input(shape=img_shape, name='image_input')

    # Use the generated model
    output = resnet_model(keras_input)

    # Add the fully-connected layers
    x = Flatten(name='flatten')(output)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    # Create your own model
    pretrained_model = Model(inputs=keras_input, outputs=x)
    pretrained_model.compile(loss='categorical_crossentropy', optimizer=optimizers.adam(lr=0.001), metrics=['accuracy'])

    return pretrained_model

# data augmentation
train_datagen = image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.1)

# generate training batches
train_generator = train_datagen.flow_from_directory(
        'data/1',
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        subset='training')

# generate validation batches
validation_generator = train_datagen.flow_from_directory(
        'data/1',
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        subset='validation')

# get classes
class_dict = train_generator.class_indices
class_dict = {v: k for k, v in class_dict.items()}

# compile model
pretrained_model = pretrained_model(train_generator.image_shape, train_generator.num_classes)

# training the model
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0)
hist = pretrained_model.fit_generator(train_generator, validation_data=validation_generator, epochs=1, callbacks=[early_stopping])

# test an image from the test set
img = image.load_img('data/test/' + '0.jpg', target_size=(224, 224))
array_img = image.img_to_array(img)
array_img = np.expand_dims(array_img, axis=0)
prediction = pretrained_model.predict(array_img)
prediction_class = prediction.argmax(axis=-1)
print(prediction_class)
print(class_dict[prediction_class[0]])