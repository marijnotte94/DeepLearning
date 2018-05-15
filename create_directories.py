import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import os

def create_dir(train_dir, target_dir, filenames, labels):
    #create target dir if not exists
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


data_dir = 'data/'
train_dir = os.path.join(data_dir, 'train')

fine_grained = 1 #years
train_target_dir = os.path.join(data_dir, str(fine_grained), 'train')
test_target_dir = os.path.join(data_dir, str(fine_grained), 'test')

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

# drop rows that have dates before 1060, these are mislabeled
df = df[df['date'] > 1059]

# drop rows that are in test set
df = df[df['in_train']]

# keep only rows that have dates that appear at least 100 times
count = pd.DataFrame(df.date.value_counts().reset_index())
count.columns = ['date', 'count']
df = pd.merge(df, count, on='date')
df = df[df['count'] > 99]

# drop artist_group and in_train columns
df = df.drop(['artist_group', 'in_train'], axis=1)

# create train and test sets
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['date'].values)
df_train = df_train.drop(['count'], axis=1)
df_test = df_test.drop(['count'], axis=1)


# create lists to hold the filenames and labels
train_filenames = list(df_train['new_filename'].values)
train_labels = list(df_train['date'].values)
print('Length train: %d' % len(train_filenames))
test_filenames = list(df_test['new_filename'].values)
test_labels = list(df_test['date'].values)
print('Length test: %d' % len(test_filenames))

# create directories for train and test
create_dir(train_dir, train_target_dir, train_filenames, train_labels)
create_dir(train_dir, test_target_dir, test_filenames, test_labels)
