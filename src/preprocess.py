import pandas as pd
import utils
from sklearn.model_selection import train_test_split
import sys
#read the file
df = pd.read_csv("/Users/deepak/Documents/OpenSource/tNASnet/dataset/AirPassengers.csv")


TRAIN_SPLIT = 100

uni_data = df['#Passengers']
uni_data.index = df['Month']
uni_data.head()

uni_data = uni_data.values

uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()

uni_data = (uni_data-uni_train_mean)/uni_train_std


# choose a number of time steps
n_steps = 10
# split into samples
X, y = utils.split_sequence(uni_data, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# train test split
# train test split
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

#print(x_train)
#print(pd.__file__)