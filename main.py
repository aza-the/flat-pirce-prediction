import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from make import *

CHECKPOINT_PATH = "trained_w/cp.ckpt"
LEARNING_RATE = 0.00006

def main(arg: str) -> None:
    if arg == 'train':
        """
            Needs a json file of data to prepare it and the traing a model.
        """
        pre_main()
        make()
        train_model()
    if arg == 'run':
        run_preditcion_on_model()


def run_preditcion_on_model(
    district: int,
    metro_name: int,
    metro_time: int,
    metro_get_type: int,
    size: int,
    kitchen: int,
    floor: int,
    floors: int,
    constructed: int,
    fix: int,
    type_of_building: int,
    type_of_walls: int,
):
    """ 
        Gets a 'list' of parameters of a flat and 
        then makes a predictions based on the trained weights.
    """

    model = build_model(LEARNING_RATE)

    model.load_weights(CHECKPOINT_PATH).expect_partial()

    mean_and_std = []
    with open(FOLDER_PATH + "/mean_and_std.txt") as file:
        mean_and_std = file.readlines()

    for i in range(len(mean_and_std)):
        mean_and_std[i] = mean_and_std[i].strip()
        mean_and_std[i] = mean_and_std[i].replace('[', '')
        mean_and_std[i] = mean_and_std[i].replace(']', '')
        mean_and_std[i] = mean_and_std[i].replace(' ', '')

    mean = mean_and_std[0].split(',')
    std = mean_and_std[1].split(',')

    mean = np.array(mean, dtype=float)
    std = np.array(std, dtype=float)

    flat = [[
        district,
        metro_name,
        metro_time,
        metro_get_type,
        size,
        kitchen,
        floor,
        floors,
        constructed,
        fix,
        type_of_building,
        type_of_walls,
    ]]

    flat = np.array(flat, dtype=float)
    flat -= mean
    flat /= std

    predictions = model.predict(flat)
    return predictions[0]


def train_model() -> None:
    """ 
        Trains model with prepared data and saves final weights for following predictions. 
    """

    flats = []
    with open(CSV_VECTORIZED_FILE_NAME) as file:
        reader = csv.reader(file)
        for row in reader:
            flats.append(np.array(row))

    # Replacing all valuse that are greater than 25 000 000 to 25 000 000.
    # That will increase the general accuracy of the model.
    count = 0
    for i in range(1, len(flats)):
        if(int(flats[i][0]) > 25000000):
            flats[i][0] = '25000000'
            count += 1
    print(f"Count: {count}")

    flats = flats[1:] # Removing first row beacuse it is not the data row
    flats = np.array(flats, dtype=float)
    np.random.shuffle(flats)
    targets = np.delete(flats, [1,2,3,4,5,6,7,8,9,10,11,12], axis=1)
    flats = np.delete(flats, [0], axis=1)

    N = len(flats)
    print(f'Length of flats[]: {N}')
    percent_80 = int(0.8 * N)

    # Slicing data on two special sets
    train_data = flats[:percent_80]
    test_data = flats[percent_80:]

    targets /= 1000000.   # Converting 17 560 000 to 17.56
    train_targets = targets[:percent_80]
    test_targets = targets[percent_80:]

    # Applying normalization
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std
    test_data -= mean
    test_data /= std

    with open(FOLDER_PATH + "/mean_and_std.txt", "w") as file:
        s = str(list(mean)) + '\n' + str(list(std))
        file.write(s)

    # Parameters for model
    epochs = 200
    batch_size = 256
    learning_rate = LEARNING_RATE

    model = build_model(learning_rate)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_weights_only=True,
        verbose=1
    )    

    history = model.fit(
        train_data,
        train_targets,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[cp_callback],
    )

    test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
    print(test_mse_score, test_mae_score)

    make_plot(history)

def build_model(learning_rate: float):
    """
        Makes a Sequential model using the given learning rate in the optimizer.
    """

    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model


def make_plot(history):
    """ 
        Makes a plot from given history. 
    """

    history_mae = history.history["val_mae"]
    epochs = range(1, len(history_mae) + 1)
    plt.plot(epochs, history_mae, "b", label="Mean Absolute Error")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1])