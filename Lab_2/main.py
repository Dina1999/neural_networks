import pandas
from keras import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plot

# Загрузка данных из файла
dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

model = Sequential()
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

train_history = model.fit(X, encoded_Y, epochs=200, batch_size=10, validation_split=0.1)

# График ошибок
plot.plot(train_history.epoch, train_history.history['loss'], label="Training Loss")
plot.plot(train_history.epoch, train_history.history['val_loss'], label="Validation Loss")
plot.xlabel('Эпохи')
plot.ylabel('Ошибки')
plot.legend()
plot.show()

# График точности
plot.clf()
plot.plot(train_history.epoch, train_history.history['accuracy'], label="Training Accuracy")
plot.plot(train_history.epoch, train_history.history['val_accuracy'], label="Validation Accuracy")
plot.xlabel('Эпохи')
plot.ylabel('Точность')
plot.legend()
plot.show()