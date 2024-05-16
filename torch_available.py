import numpy as np
import matplotlib.pyplot as plt

def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
def plot_ransac_revised(segment_data_x, segment_data_y):
    from skimage.measure import LineModelND, ransac
    data = np.column_stack([segment_data_x, segment_data_y])

    # fit line using all data
    model = LineModelND()
    model.estimate(data)

    # robustly fit line only using inlier data with RANSAC algorithm
    model_robust, inliers = ransac(data, LineModelND, min_samples=2,
                                   residual_threshold=5, max_trials=1000)
    outliers = inliers == False

    # generate coordinates of estimated models
    line_x = np.array([segment_data_x.min(), segment_data_x.max()])
    line_y = model.predict_y(line_x)
    line_y_robust = model_robust.predict_y(line_x)
    k = (line_y_robust[1] - line_y_robust[0]) / (line_x[1] - line_x[0])
    m = line_y_robust[0] - k * line_x[0]
    x0 = (segment_data_y.min() - m) / k
    x1 = (segment_data_y.max() - m) / k
    line_x_y = np.array([x0, x1])
    line_y_robust_y = model_robust.predict_y(line_x_y)
    if (distance(line_x[0], line_y_robust[0], line_x[1], line_y_robust[1]) <
            distance(line_x_y[0], line_y_robust_y[0], line_x_y[1], line_y_robust_y[1])):
                # plt.plot(line_x, line_y_robust, '-b', label='Robust line model')
        line_twopoint = (line_x, line_y_robust)
    else:
                # plt.plot(line_x_y, line_y_robust_y, '-b', label='Robust line model')
        line_twopoint = (line_x_y, line_y_robust_y)

    return inliers, outliers, line_twopoint


def line_intersection(line1, line2, x_min, x_max, y_min, y_max):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    # 범위 내의 값인지 체크
    if x_min - 100 <= x <= x_max + 100 and y_min - 100 <= y <= y_max + 100:
        return x, y
    else:
        return -12345, -12345

if __name__ == "__main__":
    whole = np.load('whole_save.npy')

    xy = np.where(whole == 1)
    x_data = xy[0]
    y_data = xy[1]
    x_tmp = x_data.copy()
    y_tmp = y_data.copy()
    ransac_line = []
    intersection_points = []
    while True:
        inliers, outliers, line_twopoint = plot_ransac_revised(x_tmp, y_tmp)

        if x_tmp[inliers].shape[0] >= 2:
            # inliers, two points for line 기록 저장
            ransac_line.append((x_tmp[inliers], y_tmp[inliers], line_twopoint))

        # 나머지 점들 (outliers)
        x_tmp = x_tmp[outliers]
        y_tmp = y_tmp[outliers]

        # if x_tmp.shape[0] <= 2 or len(ransac_line) == 4:
        if len(ransac_line) == 4:
            break
    x_min, x_max, y_min, y_max = x_data.min(), x_data.max(), y_data.min(), y_data.max()
    for i in range(len(ransac_line)):
        for j in range(i + 1, len(ransac_line)):
            (x1, x2), (y1, y2) = ransac_line[i][2]
            (x3, x4), (y3, y4) = ransac_line[j][2]
            x, y = line_intersection([[x1, y1], [x2, y2]], [[x3, y3], [x4, y4]], x_min, x_max, y_min, y_max)
            if x != -12345 or y != -12345:
                intersection_points.append(np.array((x, y)))


    intersection_points = np.array(intersection_points)
    print(intersection_points)
    plt.scatter(x_data, y_data)
    plt.scatter(intersection_points[:,0], intersection_points[:,1])
    plt.show()