import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import os
import mlflow
import mlflow.keras

target = os.environ['Stock_Name']
model_optimizer = os.environ['Compile_Optimizer']
model_loss = os.environ['Compile_Loss']
train_test_split = float(os.environ['Train_Test_Split'])
model_batch = int(os.environ['Train_Batch'])
model_epochs = int(os.environ['Train_Epochs'])
predict_target = os.environ['Predict_Target']
timestep = int(os.environ['Timestep'])

# define evaluation metric
def root_square_mean(test_data: np.ndarray, predictions: np.ndarray) -> float:
    return np.sqrt(np.mean(np.power(test_data - predictions, 2)))

# read dataset
dataset = pd.read_csv('{}.csv'.format(target))
# make sure target column's type is numeric
dataset[predict_target] = dataset[predict_target].apply(pd.to_numeric)

# feature engineering
sc = MinMaxScaler(feature_range = (0, 1))
scaled_target = sc.fit_transform(dataset[predict_target].values.reshape(-1, 1))
dataset['scaled_target'] = scaled_target

# set train and test size
training_size = int(len(dataset) * train_test_split)
test_size= len(dataset) - training_size
# split data
dataset_train ,dataset_test = dataset.iloc[0:training_size, :],dataset.iloc[training_size:len(dataset), :]

# prepare train data
training_set = dataset_train[['scaled_target']].values

## how many timesteps and 1 output
X_train = []
y_train = []
for i in range(timestep, len(training_set)):
    X_train.append(training_set[i-timestep: i, 0])
    y_train.append(training_set[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, newshape = 
                     (X_train.shape[0], X_train.shape[1], 1))

# prepare test data
testing_set = dataset_test[[predict_target]].values
testing_set.shape
dataset_total = pd.concat((dataset_train['scaled_target'], dataset_test['scaled_target']), 
                          axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
X_test = []
for i in range(timestep, len(inputs)):
    X_test.append(inputs[i-timestep:i, 0])
X_test = np.array(X_test)
#add dimension of indicator
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# build model
model = Sequential()
#add 1st lstm layer
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(rate = 0.2))

##add 2nd lstm layer: 50 neurons
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(rate = 0.2))

##add 3rd lstm layer
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(rate = 0.2))

##add 4th lstm layer
model.add(LSTM(units = 50, return_sequences = False))
model.add(Dropout(rate = 0.2))

##add output layer
model.add(Dense(units = 1))
model.summary()

# execute mlflow
with mlflow.start_run():
    model.compile(optimizer = model_optimizer, loss = model_loss, metrics=['mse', 'mae', 'mape'])
    mlflow.keras.autolog()
    results = model.fit(x = X_train, y = y_train, batch_size = model_batch, epochs = model_epochs, validation_split = 0.2)

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    r_mean_square = root_square_mean(testing_set, predicted_stock_price)
    
    # mlflow.log_param("epochs", model_epochs)
    # mlflow.log_param("batch_size", model_batch)
    mlflow.log_param("rms", r_mean_square)

    mlflow.set_tag("company", target)
    mlflow.set_tag("train_test_split", train_test_split)
    

