import pandas as pd
import shutil
import os

data_dir = 'data/'
train_dir = os.path.join(data_dir, 'train')

fine_grained = 1 #years
target_dir = os.path.join(data_dir, str(fine_grained))

# read csv with all data info
filename = 'data/all_data_info.csv'
df = pd.read_csv(filename)

# remove circas
df['date'] = df['date'].str.replace('c.', '')

# convert datetime values to only year
df['date'] = df['date'].str.replace('.*\s', '')

# drop rows that have no exact date
df = df[pd.to_numeric(df['date'], errors='coerce').notnull()]
df['date'] = pd.to_numeric(df['date'], errors='coerce', downcast='signed')

# drop rows that are in test set
df = df[df['in_train']]

# create lists to hold the filenames and labels
filenames = list(df['new_filename'].values)
labels = list(df['date'].values)
print(labels)
print(len(filenames))

# create target dir if not exists
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# copy images that are in the train set to correct bin dir
for painting, year in zip(filenames,labels):
    # create bin dir if not exists
    if not os.path.exists(os.path.join(target_dir, str(year))):
        os.makedirs(os.path.join(target_dir, str(year)))

    # copy image to correct bin dir
    painting_path = os.path.join(target_dir, str(year), painting)
    shutil.copyfile(os.path.join(train_dir, painting), painting_path)