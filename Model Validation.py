import cv2
import numpy as np
import pickle
import keras

def load_binary_image(file_path="./data/input.jpeg"):
    cimg = cv2.imread(file_path, 1)
    gimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
    gimg = cv2.medianBlur(gimg, 5)
    # threshold the image to find all points of interest
    _, bimg = cv2.threshold(gimg, 20, 255, cv2.THRESH_BINARY)

    # collect points of interest
    pts = cv2.findNonZero(bimg)
    pts = pts.reshape((-1, 2))

    return {"binary_image": bimg, "points": pts}

def recover_image(bimg, pts):
    # create canvas of same size
    canvas = np.zeros_like(bimg)

    # draw points on canvas (x, y) = (c, r)
    for pt in pts:
        try:
            canvas[pt[1], pt[0]] = 255
        except:
            pass

    # show the re-drew image and the target image as well
    cv2.imshow("canvas", canvas)
    cv2.imshow("target", bimg)
    cv2.waitKey(0)

class Restoration:
    def __init__(self, file_path="./data"):
        with open(file_path + "/geometry_data.pkl", "rb") as g_file:
            self.geometry_data = pickle.load(g_file)
        self.model = keras.models.load_model(file_path + "/pos_model.h5")

    def transform_pts(self, x_arr, round=True):
        """
        :param x_arr: the points that we want to transform from output image to input image.
        :return: y_arr, the transformed pts
        """
        model_input = (x_arr - self.geometry_data["x"]["center"]) / self.geometry_data["x"]["scale"]
        model_output = self.model.predict_on_batch(model_input)
        y_arr = model_output * self.geometry_data["y"]["scale"] + self.geometry_data["y"]["center"]
        if round:
            return y_arr.astype(int)
        else:
            return y_arr

if __name__ == "__main__":
    image_data = {"input": {}, "output": {}}
    image_data["input"] = load_binary_image(file_path="./data/input.jpeg")
    image_data["output"] = load_binary_image(file_path="./data/output.jpeg")

    restoration = Restoration()
    transformed_input_points = restoration.transform_pts(image_data["output"]["points"])
    recover_image(image_data["input"]["binary_image"], transformed_input_points)