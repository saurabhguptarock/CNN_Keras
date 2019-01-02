from pandas import read_csv
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D, Flatten, Dense, Dropout
from keras.utils import np_utils

x = read_csv('fashion-mnist_train.csv')
x_ = x.values
x = x_[:, 1:]
y = x_[:, 0]

x_train = x.reshape((-1, 28, 28, 1))
y_train = np_utils.to_categorical(y)

x = read_csv('fashion-mnist_test.csv')
x_ = x.values
x = x_[:, 1:]
y = x_[:, 0]

x_test = x.reshape((-1, 28, 28, 1))
y_test = np_utils.to_categorical(y)

model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPool2D(2, 2))
model.add(Convolution2D(32, (5, 5), activation='relu'))
model.add(Convolution2D(8, (5, 5), activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(x_train, y_train, epochs=10, shuffle=True, batch_size=256, validation_data=(x_test, y_test))
