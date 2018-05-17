import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.applications import resnet50
from sklearn.metrics import confusion_matrix
import keras.backend as K
import itertools
import os

def mean_absolute_bin_error(y_true, y_pred):
    return K.mean(K.abs(K.argmax(y_true) - K.argmax(y_pred)))

def plot_confusion_matrix(generator, predictions):
    classes = generator.class_indices.keys()

    y_true = generator.classes
    y_pred = predictions.argmax(axis=-1)

    # create normalized confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, list(classes), rotation=90)
    plt.yticks(tick_marks, list(classes))

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix_resnet50', ext='png', dpi=150)
    plt.show()

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
num_epochs = 100
num_frozen_layers = 0 # freeze first num layers, ResNet50 has 175 layers

# choose bin size
bin_size = 20 # [1, 5, 10, 20, 50, 100] years
train_dir = os.path.join('data', str(bin_size), 'train')
test_dir = os.path.join('data', str(bin_size), 'test')

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

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy_epochs_resnet50', ext='png', dpi=150)
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('loss_epochs_resnet50', ext='png', dpi=150)
plt.show()

# evaluate on test set
print('Evaluate on test set')
test_loss = pretrained_model.evaluate_generator(test_generator, use_multiprocessing=True)
print('Loss test set: ' + str(test_loss[0]))
print('Accuracy test set: ' + str(test_loss[1]))
print('Mean absolute bin error test set: ' + str(test_loss[2]))

# predict on test set
print('Predict on test set')
predictions = pretrained_model.predict_generator(test_generator, use_multiprocessing=True)

# plot confusion matrix
print('Plot confusion matrix')
plot_confusion_matrix(test_generator, predictions)
