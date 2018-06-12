from keras.preprocessing import image
from keras.models import load_model
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

# choose bin size and read train, validation and test data
bin_size = '20' # [1, 5, 10, 20, 50, 100] years
df_train = pd.read_csv(os.path.join('data', 'train.csv'), usecols=['filename', bin_size])
df_validation = pd.read_csv(os.path.join('data', 'validation.csv'), usecols=['filename', bin_size])
df_test = pd.read_csv(os.path.join('data', 'test.csv'), usecols=['filename', bin_size])
nbatches_test, mod = divmod(df_test.shape[0], 1)

# encode labels
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(df_train[bin_size])
num_classes = len(label_encoder.classes_)

#compute class weights
counter = Counter(label_encoder.transform(df_train[bin_size]))
max_val = float(max(counter.values()))
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

# dir for saving figures
figures_dir = os.path.join('figures', str(bin_size))

# load model
checkpoint_filename = 'resnet50_1e-4_false.hdf5' # change to checkpoint filename
checkpoint_dir = os.path.join('checkpoints', str(bin_size))
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
pretrained_model = load_model(checkpoint_path, custom_objects={'mean_absolute_bin_error': mean_absolute_bin_error})
print('Load model from file')
print('Learning rate: ', K.eval(pretrained_model.optimizer.lr))

# model summary
pretrained_model.summary()

# evaluate on test set
test_gen = custom_generator(df_test, label_encoder, bin_size, False, (224, 224), 1)
test_loss = pretrained_model.evaluate_generator(test_gen, steps=nbatches_test)
print('Loss test set: ' + str(test_loss[0]))
print('Accuracy test set: ' + str(test_loss[1]))
print('Mean absolute bin error test set: ' + str(test_loss[2]))

# predict on test set
test_gen = custom_generator(df_test, label_encoder, bin_size, False, (224, 224), 1)
predictions = pretrained_model.predict_generator(test_gen, steps=nbatches_test)

# plot confusion matrix
test_gen = custom_generator(df_test, label_encoder, bin_size, False, (224, 224), 1)
plots.plot_confusion_matrix(figures_dir, 'confusion_' + checkpoint_filename[:-5],
                            test_gen, df_test.shape[0], label_encoder, predictions, show_values=True)

# plot ROC curves
plots.plot_ROC(figures_dir, 'ROC_' + checkpoint_filename[:-5], test_gen, df_test.shape[0], label_encoder, predictions)