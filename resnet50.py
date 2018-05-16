import numpy as np
from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.applications import resnet50

def mean_absolute_error_years(model, generator):
    predictions = model.predict_generator(generator)
    predictions_class = predictions.argmax(axis=-1)

    mean_absolute_error = []
    for prediction in predictions_class:
        y_pred = class_dict[prediction]
        y_true = class_dict[np.argmax(test_generator.next()[1])]
        mean_absolute_error.append(abs(int(y_true) - int(y_pred)))

    return round(sum(mean_absolute_error) / len(mean_absolute_error))

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

# train and test image data generators
train_datagen = image.ImageDataGenerator(validation_split=0.1)
test_datagen = image.ImageDataGenerator()

# generate training batches
train_generator = train_datagen.flow_from_directory(
    'data/1/train',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training')

# generate validation batches
validation_generator = train_datagen.flow_from_directory(
    'data/1/train',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation')

# generate test batches
test_generator = test_datagen.flow_from_directory(
    'data/1/test',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=False)

# get classes
class_dict = test_generator.class_indices
class_dict = {v: k for k, v in class_dict.items()}

# compile model
pretrained_model = pretrained_model(train_generator.image_shape, train_generator.num_classes)

# training the model
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0)
hist = pretrained_model.fit_generator(train_generator, validation_data=validation_generator, epochs=1, callbacks=[early_stopping])

# evaluate on test set
test_loss = pretrained_model.evaluate_generator(test_generator)
print('Loss test set: ' + str(test_loss[0]) + ', Accuracy test set: ' + str(test_loss[1]))
mean_absolute_error = mean_absolute_error_years(pretrained_model, test_generator)
print('Mean absolute error in years: ' + str(mean_absolute_error))

# # test an image from the test set
# img = image.load_img('data/test/' + '0.jpg', target_size=(224, 224))
# array_img = image.img_to_array(img)
# array_img = np.expand_dims(array_img, axis=0)
# prediction = pretrained_model.predict(array_img)
# prediction_class = prediction.argmax(axis=-1)
# print(prediction_class)
# print(class_dict[prediction_class[0]])