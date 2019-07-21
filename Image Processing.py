import cv2
import numpy as np

def get_centers(file_path = "./data/input.jpeg", n_rows=10, n_cols=10, verbose=False, image_thresh=20, hough_thresh=20, minDist=20, maxRadius=20):
    """
    return an array of shape 100, 3. for each row, the first two col represent x, y of the circle, the third col represent the avg intensity of the circle.
    """
    print("looking for circles in the image")
    cimg = cv2.imread(file_path, 1)
    gimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
    gimg = cv2.medianBlur(gimg, 5)
    # threshold the image for better circle finding
    _, gimg = cv2.threshold(gimg, image_thresh, 255, cv2.THRESH_BINARY)


    # find circles on the image
    circles = cv2.HoughCircles(gimg, cv2.HOUGH_GRADIENT, dp=1, minDist=minDist, param1=1, param2=hough_thresh, minRadius=5,
                               maxRadius=maxRadius)

    # re-sequence the circles found
    circles = circles[0, :, :].reshape((-1, 3))
    x_min = circles.min(axis=0)[0]
    x_max = circles.max(axis=0)[0]
    avg_spacing = (x_max - x_min) / (n_cols - 1)
    x_segment = [x_min - avg_spacing / 2 + avg_spacing * i for i in range(11)]
    def get_col_id(point):
        # return which col this point sits
        x = point[0]
        for i in range(n_cols):
            if x_segment[i] < x < x_segment[i + 1]:
                return i
    cols = [[] for i in range(n_cols)]
    for circle in circles:
        id = get_col_id(circle)
        cols[get_col_id(circle)].append(circle)
    for i, col in enumerate(cols):
        cols[i] = list(sorted(col, key=lambda p:p[1]))
    grid = np.zeros((n_rows, n_cols, 3))
    for c in range(n_cols):
        for r in range(n_rows):
            grid[r, c, :] = cols[c][r]

    # plot theses circles to make sure the sequence is correct
    gaussian_img = cv2.GaussianBlur(cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    print("show the circles found")
    for c in range(10):
        for r in range(10):
            pos = grid[r, c, :2].astype(np.int)
            radius = grid[r, c, 2].astype(np.int)
            # draw the outer circle
            cv2.circle(cimg, (pos[0], pos[1]), radius, (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (pos[0], pos[1]), 2, (0, 0, 255), 3)
            # find the avg intensity at each center
            grid[r, c, 2] = gaussian_img[pos[1], pos[0]]
            if verbose:
                cv2.imshow('circles found in the image', cimg)
                cv2.waitKey(10)

    if verbose:
        cv2.imshow('circles found in the image', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return grid

def recenter_grid(grid):
    center_of_centers = grid.mean(axis=0).mean(axis=0).reshape((1, 1, 3))
    center_of_centers[:, :, 2] = 0
    recentered_grid = grid - center_of_centers
    return recentered_grid

def normalize_grid(grid):
    # get the recentered data set
    grid = recenter_grid(grid)

    # scale the intensity to range from 0 - 255 to 0 - 1
    grid[:, :, 2] = grid[:, :, 2] / 255

    # scale the positions such that the 0, 0 point and 9, 9 point have distance of 1
    vec = grid[0, 0, :2] - grid[-1, -1, :2]
    length = np.linalg.norm(vec, ord=2)
    grid[:, :, :2] = grid[:, :, :2] / length

    return grid

if __name__ == "__main__":
    n_cols = 10
    n_rows = 10

    # get the positions on the image
    input_grid = get_centers(file_path="./data/input.jpeg", n_cols=n_cols, n_rows=n_rows, verbose=False, hough_thresh=20, minDist=20, maxRadius=20)
    output_grid = get_centers(file_path="./data/output.jpeg", n_cols=n_cols, n_rows=n_rows, verbose=False, hough_thresh=10, minDist=50, maxRadius=40)

    # normalize the positions
    input_grid = normalize_grid(input_grid)
    output_grid = normalize_grid(output_grid)

    # construct the input and output of the function learning
    x = np.zeros((n_cols * n_rows, 3))
    y = np.zeros((n_cols * n_rows, 3))
    for r in range(n_rows):
        for c in range(n_cols):
            # since we are trying to learn the inverse projection, we set output_grid to x, but input_grid to y
            x[r * n_cols + c, :] = output_grid[r, c, :]
            y[r * n_cols + c, :] = input_grid[r, c, :]
    np.savez("./data/train_data.npz", x=x, y=y)


