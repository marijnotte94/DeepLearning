from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.applications import resnet50
import keras.backend as K
import os
import plots

def mean_absolute_bin_error(y_true, y_pred):
    return K.mean(K.abs(K.argmax(y_true) - K.argmax(y_pred)))

def pretrained_model(img_shape, num_classes, learning_rate, num_frozen_layers):
    # import model pretrained on imagenet without last layer
    resnet_model = resnet50.ResNet50(include_top=False, weights='imagenet')

    # freeze layers
    for layer in resnet_model.layers[:num_frozen_layers]:
        layer.trainable = False

    # input shape
    keras_input = Input(shape=img_shape, name='image_input')

    # use the generated model
    output = resnet_model(keras_input)

    # add the fully-connected layers
    x = Flatten(name='flatten')(output)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    # compile model
    pretrained_model = Model(inputs=keras_input, outputs=x)
    pretrained_model.compile(loss='categorical_crossentropy',
                             optimizer=optimizers.adam(lr=learning_rate),
                             metrics=['accuracy', mean_absolute_bin_error])

    return pretrained_model

# hyperparameters
learning_rate = 0.001
batch_size = 16
num_epochs = 1
num_frozen_layers = 175 # freeze first num layers, ResNet50 has 175 layers

# choose bin size
bin_size = 20 # [1, 5, 10, 20, 50, 100] years
train_dir = os.path.join('data', str(bin_size), 'train')
test_dir = os.path.join('data', str(bin_size), 'test')
figures_dir = os.path.join('figures', str(bin_size))

# train and test image data generators
train_datagen = image.ImageDataGenerator(validation_split=0.1)
test_datagen = image.ImageDataGenerator()

# generate training batches
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

# generate validation batches
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# generate test batches
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# compile model
pretrained_model = pretrained_model(train_generator.image_shape, train_generator.num_classes,
                                    learning_rate, num_frozen_layers)

# model summary
pretrained_model.summary()

# training the model
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2)
history = pretrained_model.fit_generator(train_generator, validation_data=validation_generator,
                                         epochs=num_epochs, callbacks=[early_stopping])

# plot accuracy and loss for train and validation
if num_epochs > 1:
    plots.plot_accuracy(figures_dir, 'accuracy_epochs_resnet50', history)
    plots.plot_loss(figures_dir, 'loss_epochs_resnet50', history)

# evaluate on test set
test_loss = pretrained_model.evaluate_generator(test_generator, use_multiprocessing=True)
print('Loss test set: ' + str(test_loss[0]))
print('Accuracy test set: ' + str(test_loss[1]))
print('Mean absolute bin error test set: ' + str(test_loss[2]))

# predict on test set
predictions = pretrained_model.predict_generator(test_generator, use_multiprocessing=True)

# plot confusion matrix
plots.plot_confusion_matrix(figures_dir, 'confusion_matrix_resnet50', test_generator, predictions, show_values=True)
