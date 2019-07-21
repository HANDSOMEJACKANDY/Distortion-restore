import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD, Adam

if __name__ == "__main__":
    # get data
    train_data = np.load("./data/train_data.npz")
    train_data = dict(train_data)
    x = train_data["x"]
    y = train_data["y"]

    # position data
    pos_x = x[:, :2]
    pos_y = y[:, :2]
    intensity_ratio = y[:, 2] / x[:, 2]

    # find mean square error of identity projection
    identity_mse = np.square(pos_x - pos_y).mean()
    intensity_mse = np.square(intensity_ratio - intensity_ratio.mean()).mean()
    print("mse for identity projection: ", identity_mse)
    print("mse for mean intensity prediction: ", intensity_mse)
    input()

    # formulate position model
    pos_input = Input(shape=(2,))
    intermediate_embedding = Dense(units=100, activation="tanh", use_bias=True)(pos_input)
    pos_output = Dense(units=2, activation="tanh", use_bias=True)(intermediate_embedding)

    # formulate intensity model
    intensity_output = Dense(units=1, activation="sigmoid", use_bias=True)(intermediate_embedding)

    model = Model(inputs=[pos_input], outputs=[pos_output, intensity_output])
    optimiser = Adam(lr=0.001)
    model.compile(optimizer=optimiser, loss='mse')

    model.fit(x=pos_x, y=[pos_y, intensity_ratio], epochs=100000, batch_size=90, validation_split=0.1, verbose=1)
