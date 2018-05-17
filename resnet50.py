import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.applications import resnet50
from sklearn.metrics import confusion_matrix

def mean_absolute_error_years(generator, predictions):
    # get class labels
    class_dict = generator.class_indices
    class_dict = {v: k for k, v in class_dict.items()}

    # get predicted classes
    predictions_class = predictions.argmax(axis=-1)

    # calculate mean absolute error in years
    mean_absolute_error = []
    for prediction in predictions_class:
        y_pred = class_dict[prediction]
        y_true = class_dict[np.argmax(generator.next()[1])]
        mean_absolute_error.append(abs(int(y_true) - int(y_pred)))

    return round(sum(mean_absolute_error) / len(mean_absolute_error))

def plot_confusion_matrix(generator, predictions):
    # get class labels
    class_dict = generator.class_indices
    class_dict = {v: k for k, v in class_dict.items()}

    # get predicted classes
    predictions_class = predictions.argmax(axis=-1)

    # create lists with predictions and labels
    y_true = []
    y_pred = []
    for prediction in predictions_class:
        y_pred.append(class_dict[prediction])
        y_true.append(class_dict[np.argmax(generator.next()[1])])

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(15, 15))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_dict))
    plt.xticks(tick_marks, list(class_dict.values()), rotation=90)
    plt.yticks(tick_marks, list(class_dict.values()))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix', ext='png', dpi=150)
    plt.show()

def pretrained_model(img_shape, num_classes):
    # import model pretrained on imagenet without last layer
    resnet_model = resnet50.ResNet50(include_top=False, weights='imagenet')
    for layer in resnet_model.layers:
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

# compile model
pretrained_model = pretrained_model(train_generator.image_shape, train_generator.num_classes)

# training the model
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2)
history = pretrained_model.fit_generator(train_generator, validation_data=validation_generator, epochs=10, callbacks=[early_stopping])

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# evaluate loss and accuracy on test set
test_loss = pretrained_model.evaluate_generator(test_generator)
print('Loss test set: ' + str(test_loss[0]) + ', Accuracy test set: ' + str(test_loss[1]))

# predict on test set
predictions = pretrained_model.predict_generator(test_generator)

# mean absolute error in years
mean_absolute_error = mean_absolute_error_years(test_generator, predictions)
print('Mean absolute error in years: ' + str(mean_absolute_error))

# plot confusion matrix
plot_confusion_matrix(test_generator, predictions)
