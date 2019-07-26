import numpy as np
import pickle
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD, Adam
import sys

class Distortion:
    def __init__(self, file_path="./data"):
        # load training point data
        self.original_data = {}
        self.file_path = file_path
        saved_data = np.load(self.file_path + "/train_data.npz")
        self.original_data["original_x"] = saved_data["x"]
        self.original_data["original_y"] = saved_data["y"]

        # flatten the data
        n_rows, n_cols, _ = self.original_data["original_x"].shape
        self.original_data["flat_x"] = np.zeros((n_rows * n_cols, 3))
        self.original_data["flat_y"] = np.zeros((n_rows * n_cols, 3))
        for r in range(n_rows):
            for c in range(n_cols):
                # since we are trying to learn the inverse projection, we set output_grid to x, but input_grid to y
                self.original_data["flat_x"][r * n_cols + c, :] = self.original_data["original_x"][r, c, :]
                self.original_data["flat_y"][r * n_cols + c, :] = self.original_data["original_y"][r, c, :]

        # get center and scale of x, y position
        self.geometry_data = {"x": {}, "y": {}}

        # get boundary ((x_min, y_max), (x_max, y_min))
        self.geometry_data["x"]["boundary"] = (tuple(self.original_data["flat_x"].min(axis=0)[:2]), tuple(self.original_data["flat_x"].max(axis=0)[:2]))
        self.geometry_data["y"]["boundary"] = (tuple(self.original_data["flat_y"].min(axis=0)[:2]), tuple(self.original_data["flat_y"].max(axis=0)[:2]))

        # get scale: the division factor for x, y to reach the same size
        self.geometry_data["x"]["scale"] = np.linalg.norm(self.original_data["original_x"][0, 0, :2] - self.original_data["original_x"][-1, -1, :2], ord=2)
        self.geometry_data["y"]["scale"] = np.linalg.norm(self.original_data["original_y"][0, 0, :2] - self.original_data["original_y"][-1, -1, :2], ord=2)

        # get center
        self.geometry_data["x"]["center"] = self.original_data["flat_x"][:, :2].mean(axis=0).reshape((1, 2))
        self.geometry_data["y"]["center"] = self.original_data["flat_y"][:, :2].mean(axis=0).reshape((1, 2))

        # get training data
        self.positive_train_data = {}
        self.positive_train_data["x"] = self.norm_pos_arr(self.original_data["flat_x"][:, :2], pos_type="x")
        self.positive_train_data["y"] = self.norm_pos_arr(self.original_data["flat_y"][:, :2], pos_type="y")
        self.positive_train_data["intensity_ratio"] = self.original_data["flat_y"][:, 2] / self.original_data["flat_x"][:, 2]

        # get error for trivial transformation
        self.positive_train_data["pos_error"] = np.square(self.positive_train_data["y"] - self.positive_train_data["x"]).mean()
        self.positive_train_data["intensity_error"] = np.square(self.positive_train_data["intensity_ratio"] - self.positive_train_data["intensity_ratio"].mean()).mean()
        print("baseline position transformation mse error is:  ", self.positive_train_data["pos_error"])
        print("baseline intensity transformation mse error is: ", self.positive_train_data["intensity_error"])

        # create position model
        self.create_model()

    def norm_pos_arr(self, pos_arr, pos_type="x"):
        return (pos_arr - self.geometry_data[pos_type]["center"]) / self.geometry_data[pos_type]["scale"]

    def transform_pos_x2y(self, x_arr, model):
        model_input = (x_arr - self.geometry_data["x"]["center"] ) / self.geometry_data["x"]["scale"]
        model_output = model_input
        y_arr = (model_output + self.geometry_data["y"]["center"]) * self.geometry_data["y"]["scale"]
        return y_arr

    def generate_points(self, pos_type="x", delta=0.05, n=100, norm=True):
        '''
        The reg training sample is useless when the learning speed is low enough
        '''
        # get tolerance going across the bound
        delta_x = (self.geometry_data[pos_type]["boundary"][1][0] - self.geometry_data[pos_type]["boundary"][0][0]) * delta
        delta_y = (self.geometry_data[pos_type]["boundary"][1][1] - self.geometry_data[pos_type]["boundary"][0][1]) * delta

        # get applied x, y boundary
        x_min = self.geometry_data[pos_type]["boundary"][0][0] - delta_x
        x_span = self.geometry_data[pos_type]["boundary"][1][0] + delta_x - x_min
        y_min = self.geometry_data[pos_type]["boundary"][0][1] - delta_y
        y_span = self.geometry_data[pos_type]["boundary"][1][1] + delta_y - y_min
        span = np.array((x_span, y_span)).reshape((1, 2))
        start = np.array((x_min, y_min)).reshape((1, 2))

        # generate an array of random points in the boundary
        random_points = span * np.random.random((n, 2)) + start

        # normalize the points if required
        if norm:
            random_points = self.norm_pos_arr(random_points, pos_type=pos_type)

        return random_points

    def generate_regularized_pos_training_data(self, validation_split=0.1, reg_sample_ratio=10, reg_rate=0.1):
        n_validation_split = int(self.positive_train_data["x"].shape[0] * (1 - validation_split))
        n_reg_samples = int(reg_sample_ratio * n_validation_split)
        reg_sample_weight = reg_rate / n_reg_samples

        # container for output
        output = {}

        # get training data set
        combined_positive_samples = np.concatenate([self.positive_train_data["x"], self.positive_train_data["y"]], axis=1)
        np.random.shuffle(combined_positive_samples)
        combined_train_samples = combined_positive_samples[:n_validation_split, :]
        combined_validation_samples = combined_positive_samples[n_validation_split:, :]


        # get regularization samples
        reg_samples = self.generate_points(pos_type="x", delta=0.05, n=n_reg_samples, norm=True)
        reg_samples = np.concatenate([reg_samples, reg_samples], axis=1)
        combined_train_samples = np.concatenate([combined_train_samples, reg_samples], axis=0)

        # create sample weight
        sample_weights = np.ones(n_reg_samples + n_validation_split) * reg_sample_weight
        sample_weights[:n_validation_split] = 1

        output["sample_weights"] = sample_weights
        output["validation_samples"] = [combined_validation_samples[:, :2], combined_validation_samples[:, 2:]]
        output["train_samples"] = [combined_train_samples[:, :2], combined_train_samples[:, 2:]]

        return output

    def create_model(self):
        # formulate position model
        pos_input = Input(shape=(2,))
        intermediate_embedding = Dense(units=100, activation="tanh", use_bias=True)(pos_input)
        pos_output = Dense(units=2, activation="tanh", use_bias=True)(intermediate_embedding)
        self.model = Model(inputs=[pos_input], outputs=[pos_output])

    def train_model(self, iteration=200000, reg_rate=0, lr=0.00001):
        # prepare optimiser
        optimiser = Adam(lr=lr)
        self.model.compile(optimizer=optimiser, loss='mse')

        # prepare train data
        if reg_rate == 0:
            train_data = self.generate_regularized_pos_training_data(validation_split=0.1, reg_sample_ratio=10, reg_rate=reg_rate)
        else:
            train_data = self.generate_regularized_pos_training_data(validation_split=0.1, reg_sample_ratio=10, reg_rate=reg_rate)

        # start training
        for i in range(iteration):
            train_loss = self.model.train_on_batch(train_data["train_samples"][0], train_data["train_samples"][1], sample_weight=train_data["sample_weights"])
            validation_loss = self.model.test_on_batch(train_data["validation_samples"][0], train_data["validation_samples"][1])
            random_points = self.generate_points("x", n=1000, norm=True)
            deviation_from_identity = self.model.test_on_batch(random_points, random_points)
            if i % 100 == 0:
                sys.stdout.write(
                    "\rtraining iter no. {0}, training_loss: {1}, val_loss: {2}, deviation_from_identity: {3}".
                    format(i, train_loss, validation_loss, deviation_from_identity / self.positive_train_data["pos_error"]))
                sys.stdout.flush()
        print("\ntraining finish")

    def save_param(self):
        # save the pos model
        self.model.save(self.file_path + "/pos_model.h5")

        # save transformation param
        with open(self.file_path + "/geometry_data.pkl", "wb") as g_file:
            pickle.dump(self.geometry_data, g_file)



if __name__ == "__main__":
    # get data
    distortion = Distortion()
    distortion.train_model(reg_rate=0, iteration=200000, lr=0.00001)
    distortion.save_param()

    # d = {}
    # with open("./data/geometry_data.pkl", "rb") as f:
    #     d = pickle.load(f)
    #     print(d)
