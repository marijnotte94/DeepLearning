import pandas as pd
import shutil
import os

train_dir = 'data/train/'
train_filtered_dir = 'data/train_filtered/'
if not os.path.exists(train_filtered_dir):
    os.makedirs(train_filtered_dir)

# read csv with all data info
filename = 'data/all_data_info.csv'
df = pd.read_csv(filename)

# drop rows that have no date
df.dropna(axis=0, subset=['date'], inplace=True)

# drop rows that have no exact date
date = df['date'].astype('str')
mask = (date.str.len() <= 4)
df = df.loc[mask]

# drop rows that are in test set
df = df[df['in_train']]

filenames = list(df['new_filename'].values)
labels = list(df['date'].values)

# copy images that are in the filtered train set to another folder
for filename in filenames:
    shutil.copyfile((train_dir + filename), (train_filtered_dir + filename))
