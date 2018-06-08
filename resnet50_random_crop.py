from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense
from keras.models import Model, load_model
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications import resnet50
from keras.utils import to_categorical
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import os
import plots

def random_crop(image, crop_size):
    height = image.shape[0]
    width = image.shape[1]
    dy, dx = crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)

    return image[y:(y+dy), x:(x+dx), :]

def center_crop(image, crop_size):
    dy, dx = crop_size
    midpoint = np.round([image.shape[0] / 2, image.shape[1] / 2], 0)
    H = np.array([midpoint[0] - (dy / 2), midpoint[0] + (dy / 2)]).astype(int)
    W = np.array([midpoint[1] - (dx / 2), midpoint[1] + (dx / 2)]).astype(int)

    return image[H[0]:H[1], W[0]:W[1], :]

def custom_generator(df, label_encoder, bin_size, random, crop_size, batch_size):
    i = 0
    while True:
        image_batch = []
        labels_batch = []
        for j in range(batch_size):

            # shuffle data after each epoch
            if i == df.shape[0]:
                i = 0
                df = shuffle(df)

            # get filename of sample
            sample = df.iloc[i]
            filename = 'data/train/' + sample['filename']

            # one-hot encode label
            label = sample[bin_size]
            encoded_label = label_encoder.transform([label])
            dummy_label = to_categorical(encoded_label, num_classes=len(label_encoder.classes_)).flatten()

            # load image
            try:
                img = image.img_to_array(image.load_img(filename))
            except:
                return

            # random or center crop
            if random == True:
                cropped_img = random_crop(img, crop_size)

                # horizontal flip with a 1 in 2 chance
                if np.random.rand() > .5:
                    cropped_img = np.fliplr(cropped_img)
            else:
                cropped_img = center_crop(img, crop_size)

            # append to lists
            image_batch.append(cropped_img)
            labels_batch.append(dummy_label)

            i += 1

            # save image to disk
            #im = Image.fromarray(cropped_img.astype('uint8'))
            #im.save('data/images/' + sample['filename'])

        yield(np.array(image_batch), np.array(labels_batch))

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
learning_rate = 0.0001
batch_size = 16
num_epochs = 1
num_frozen_layers = 0 # freeze first num layers, ResNet50 has 175 layers
save_checkpoint = False
continue_from_checkpoint = False
checkpoint_filename = 'resnet50_best-05-0.59.hdf5' # change to checkpoint filename

# choose bin size and read train, validation and test data
bin_size = '100' # [1, 5, 10, 20, 50, 100] years
df_train = pd.read_csv(os.path.join('data', 'train.csv'), usecols=['filename', bin_size])
df_validation = pd.read_csv(os.path.join('data', 'validation.csv'), usecols=['filename', bin_size])
df_test = pd.read_csv(os.path.join('data', 'test.csv'), usecols=['filename', bin_size])
nbatches_train, mod = divmod(df_train.shape[0], batch_size)
nbatches_validation, mod = divmod(df_validation.shape[0], batch_size)
nbatches_test, mod = divmod(df_test.shape[0], 1)

# encode labels
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(df_train[bin_size])
num_classes = len(label_encoder.classes_)

# generators
train_gen = custom_generator(df_train, label_encoder, bin_size, True, (224, 224), batch_size)
validation_gen = custom_generator(df_validation, label_encoder, bin_size, False, (224, 224), batch_size)
test_gen = custom_generator(df_test, label_encoder, bin_size, False, (224, 224), 1)

# dir for saving figures
figures_dir = os.path.join('figures', str(bin_size))

# compile model
pretrained_model = pretrained_model((224, 224, 3), num_classes, learning_rate, num_frozen_layers)

# model summary
pretrained_model.summary()

# training the model
early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=2)
history = pretrained_model.fit_generator(train_gen, steps_per_epoch=nbatches_train,
                                         validation_data=validation_gen, validation_steps=nbatches_validation,
                                         epochs=num_epochs)

# plot accuracy and loss for train and validation
if num_epochs > 1:
    plots.plot_accuracy(figures_dir, 'accuracy_epochs_resnet50', history)
    plots.plot_loss(figures_dir, 'loss_epochs_resnet50', history)

# evaluate on test set
test_loss = pretrained_model.evaluate_generator(test_gen, steps=nbatches_test)
print('Loss test set: ' + str(test_loss[0]))
print('Accuracy test set: ' + str(test_loss[1]))
print('Mean absolute bin error test set: ' + str(test_loss[2]))

# predict on test set
predictions = pretrained_model.predict_generator(test_gen, steps=nbatches_test)

test_gen = custom_generator(df_test, label_encoder, bin_size, False, (224, 224), 1)
# plot confusion matrix
plots.plot_confusion_matrix(figures_dir, 'confusion_matrix_resnet50',
                            test_gen, df_test.shape[0], label_encoder, predictions, show_values=True)
