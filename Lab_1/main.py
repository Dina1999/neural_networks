import pandas
from keras import Sequential
from keras.src.layers import Dense
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plot

# Загрузка данных из файла
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

# Преобразуем данные в форму, подходящую для работы в библиотеках нейронных сетей (тензоры)
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)
dummy_y = to_categorical(encoded_Y)

# Создание модели нейронной сети
model = Sequential()
model.add(Dense(4, activation="relu"))
model.add(Dense(3, activation="softmax"))

# Параметры обучения
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение сети
train_history = model.fit(X, dummy_y, epochs=100, batch_size=10, validation_split=0.1)


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