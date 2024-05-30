import numpy as np
from keras import Sequential
from keras.src.datasets import boston_housing
from keras.src.layers import Dense
import matplotlib.pyplot as plot


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

k = 5
num_val_samples = len(train_data) // k
num_epochs = 40
all_scores = []
all_val_loss = []
all_val_mae = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    train_history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0, validation_data=(val_data, val_targets))
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    all_val_loss.append(train_history.history['val_loss'])
    all_val_mae.append(train_history.history['val_mae'])
    print(np.mean(all_scores))

    fig, ax = plot.subplots(2, 2)
    ax[0][0].plot(train_history.epoch, train_history.history['loss'], label="Training Loss")
    ax[0][0].plot(train_history.epoch, train_history.history['val_loss'], label="Validation Loss")
    ax[0][0].set_title(f'Loss (k = {i})')
    ax[0][0].set_xlabel("Эпохи")
    ax[0][0].set_ylabel("Ошибки")
    ax[0][0].legend()

    ax[0][1].plot(train_history.epoch, train_history.history['mae'], label="Training MAE")
    ax[0][1].plot(train_history.epoch, train_history.history['val_mae'], label="Validation MAE")
    ax[0][1].set_title(f'Средняя абсолютная ошибка MAE')
    ax[0][1].set_xlabel("Эпохи")
    ax[0][1].set_ylabel("MAE")
    ax[0][1].legend()
    plot.show()

plot.plot(range(1, num_epochs + 1), np.asarray(all_val_loss).mean(axis=0))
plot.title('Среднее значение потерь')
plot.xlabel('Эпохи')
plot.ylabel('Потери')
plot.show()

plot.plot(range(1, num_epochs + 1), np.asarray(all_val_mae).mean(axis=0))
plot.title('Среднее значение MAE')
plot.xlabel('Эпохи')
plot.ylabel('Потери')
plot.show()