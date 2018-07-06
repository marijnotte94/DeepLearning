import pandas as pd
from sklearn.model_selection import train_test_split
import os

data_dir = 'data/'
train_dir = os.path.join(data_dir, 'train')

def train_target_dir(bin_size):
    return os.path.join(data_dir, str(bin_size), 'train')

# read csv with all data info
filename = 'data/all_data_info.csv'
df = pd.read_csv(filename)

# drop rows that are in test set
df = df[df['in_train']]

# remove circas
df['date'] = df['date'].str.replace('c.', '')

# convert datetime values to only year
df['date'] = df['date'].str.replace('.*\s', '')

# drop rows that have no exact date
df = df[pd.to_numeric(df['date'], errors='coerce').notnull()]
df['date'] = pd.to_numeric(df['date'], errors='coerce', downcast='signed')

# drop rows that have dates before 1060, these are mislabeled
df = df[df['date'] > 1059]

# keep only rows that have dates that appear at least 199 times
count = pd.DataFrame(df.date.value_counts().reset_index())
count.columns = ['date', 'count']
df = pd.merge(df, count, on='date')
df = df[df['count'] > 199]

# drop artist_group and in_train columns
df = df.drop(['artist_group', 'in_train'], axis=1)

# create columns with different date resolutions
def add_label(year, bin_size):
    label = str(divmod(year, bin_size)[0]*bin_size) + "-" + str((1 + divmod(year, bin_size)[0])*bin_size-1)
    return label

df['1'] = df.date
df['5'] = df.date.apply(add_label,bin_size=5)
df['10'] = df.date.apply(add_label,bin_size=10)
df['20'] = df.date.apply(add_label,bin_size=20)
df['50'] = df.date.apply(add_label,bin_size=50)
df['100'] = df.date.apply(add_label,bin_size=100)

# drop unnecessary columns
df = df.drop(['source', 'pixelsx', 'pixelsy', 'size_bytes', 'count'], axis=1)
df.rename(columns=({'new_filename': 'filename'}), inplace=True)

# create train, validation and test sets
print('Total samples: ', df.shape[0])
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['date'].values)
df_train, df_validation = train_test_split(df_train, test_size=0.1, random_state=42, stratify=df_train['date'].values)

# create csv files
df_train.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
df_validation.to_csv(os.path.join(data_dir, 'validation.csv'), index=False)
df_test.to_csv(os.path.join(data_dir, 'test.csv'), index=False)