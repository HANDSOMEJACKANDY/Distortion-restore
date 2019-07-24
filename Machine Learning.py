import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD, Adam

class Distortion:
    def __init__(self, file_path="./data/train_data.npz"):
        # load training point data
        self.original_data = {}
        saved_data = np.load("./data/train_data.npz")
        self.original_data["original_x"] = saved_data["x"]
        self.original_data["original_y"] = saved_data["y"]

        # flatten the data
        n_rows, n_cols, _ = tuple(self.original_data["original_x"].shape())
        self.original_data["flat_x"] = np.zeros((n_rows * n_cols, 3))
        self.original_data["flat_y"] = np.zeros((n_rows * n_cols, 3))
        for r in range(n_rows):
            for c in range(n_cols):
                # since we are trying to learn the inverse projection, we set output_grid to x, but input_grid to y
                self.original_data["flat_x"][r * n_cols + c, :] = self.original_data["original_x"][r, c, :]
                self.original_data["flat_y"][r * n_cols + c, :] = self.original_data["original_y"][r, c, :]

        # get center and scale of x, y position
        self.geometry_data = {"x": {}, "y": {}}
        # get scale: the division factor for x, y to reach the same size
        self.geometry_data["x"]["scale"] = np.linalg.norm(self.original_data["original_x"][0, 0, :2] - self.original_data["original_x"][-1, -1, :2], ord=2)
        self.geometry_data["y"]["scale"] = np.linalg.norm(self.original_data["original_y"][0, 0, :2] - self.original_data["original_y"][-1, -1, :2], ord=2)

        # get center
        self.geometry_data["x"]["center"] = self.original_data["flat_x"].mean(axis=0).reshape((1, 2))
        self.geometry_data["y"]["center"] = self.original_data["flat_y"].mean(axis=0).reshape((1, 2))

        # get training data
        self.positive_train_data = {}
        self.positive_train_data["x"] = self.norm_pos_arr(self.original_data["flat_x"][:, :2], pos_type="x")
        self.positive_train_data["y"] = self.norm_pos_arr(self.original_data["flat_y"][:, :2], pos_type="y")
        self.positive_train_data["intensity_ratio"] = y[:, 2] / x[:, 2]

        # get error for trivial transformation
        self.positive_train_data["pos_error"] = np.square(self.positive_train_data["y"] - self.positive_train_data["x"]).mean()
        self.positive_train_data["intensity_error"] = np.square(self.positive_train_data["intensity_ratio"] - self.positive_train_data["intensity_ratio"].mean()).mean()

    def norm_pos_arr(self, pos_arr, pos_type="x"):
        return (pos_arr - self.geometry_data[pos_type]["center"]) / self.geometry_data[pos_type]["scale"]

    def transform_x2y(self, x_arr, model):
        model_input = (x_arr - self.geometry_data["x"]["center"] ) / self.geometry_data["x"]["scale"]
        model_output = model_input
        y_arr = (model_output + self.geometry_data["y"]["center"]) * self.geometry_data["y"]["scale"]
        return y_arr




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
