from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense
from keras.models import Model, load_model
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications import vgg16
from keras.utils import to_categorical
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from collections import Counter
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
    vgg_model = vgg16.VGG16(include_top=False, weights='imagenet')
    vgg_full = vgg16.VGG16(include_top=True, weights='imagenet')

    # extract last layers
    fc1_layer = vgg_full.get_layer('fc1')
    fc2_layer = vgg_full.get_layer('fc2')

    # freeze layers
    for layer in vgg_model.layers[:num_frozen_layers]:
        layer.trainable = False

    # input shape
    keras_input = Input(shape=img_shape, name='image_input')

    # use the generated model
    output = vgg_model(keras_input)

    # add the fully-connected layers
    x = Flatten(name='flatten')(output)
    #x = Dense(1024, activation='relu', name='fc6')(x)
    #x = Dense(1024, activation='relu', name='fc7')(x)
    x = fc1_layer(x)
    x = fc2_layer(x)
    x = Dense(num_classes, activation='softmax', name='fc8')(x)

    # compile model
    pretrained_model = Model(inputs=keras_input, outputs=x)
    pretrained_model.compile(loss='categorical_crossentropy',
                             optimizer=optimizers.adam(lr=learning_rate),
                             metrics=['accuracy', mean_absolute_bin_error])

    return pretrained_model

# hyperparameters
learning_rate = 0.0001
batch_size = 64
num_epochs = 100
num_frozen_layers = 19 # freeze first num layers, VGG16 has 19 layers
save_checkpoint = True
use_class_weights = True

# choose bin size and read train, validation and test data
bin_size = '20' # [1, 5, 10, 20, 50, 100] years
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

#compute class weights
counter = Counter(label_encoder.transform(df_train[bin_size]))
max_val = float(max(counter.values()))
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

# train and validation generators
train_gen = custom_generator(df_train, label_encoder, bin_size, True, (224, 224), batch_size)
validation_gen = custom_generator(df_validation, label_encoder, bin_size, False, (224, 224), batch_size)

# dir for saving figures
figures_dir = os.path.join('figures', str(bin_size))

# compile model
pretrained_model = pretrained_model((224, 224, 3), num_classes, learning_rate, num_frozen_layers)

# model summary
pretrained_model.summary()

# keep only a single checkpoint, the best over validation accuracy.
checkpoint_dir = os.path.join('checkpoints', str(bin_size))
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint = ModelCheckpoint('{0}/vgg16_best.hdf5'.format(checkpoint_dir),
                             monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# training the model
early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=2)
if save_checkpoint:
    if use_class_weights:
        history = pretrained_model.fit_generator(train_gen, steps_per_epoch=nbatches_train, epochs=num_epochs,
                                                class_weight=class_weights,
                                                validation_data=validation_gen, validation_steps=nbatches_validation,
                                                callbacks=[early_stopping, checkpoint])
    else:
        history = pretrained_model.fit_generator(train_gen, steps_per_epoch=nbatches_train, epochs=num_epochs,
                                                validation_data=validation_gen, validation_steps=nbatches_validation,
                                                callbacks=[early_stopping, checkpoint])
else:
    if use_class_weights:
        history = pretrained_model.fit_generator(train_gen, steps_per_epoch=nbatches_train, epochs=num_epochs,
                                                class_weight=class_weights,
                                                validation_data=validation_gen, validation_steps=nbatches_validation,
                                                callbacks=[early_stopping])
    else:
        history = pretrained_model.fit_generator(train_gen, steps_per_epoch=nbatches_train, epochs=num_epochs,
                                                validation_data=validation_gen, validation_steps=nbatches_validation,
                                                callbacks=[early_stopping])

# plot accuracy and loss for train and validation
if num_epochs > 1:
    plots.plot_accuracy(figures_dir, 'accuracy_epochs_vgg16', history)
    plots.plot_loss(figures_dir, 'loss_epochs_vgg16', history)