import cv2
import matplotlib.pyplot as plt
import numpy as np
def EIFS_automation(src, coord):
    def sizing(src, coord):
        data_size = np.shape(coord)
        img = cv2.imread(src)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.resize(img, data_size)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img
    def seek(coord, b=int(1.0)):
        point = np.where(coord == b)
        point_x = point[0]
        point_y = point[1]
        point_xy = np.stack((point_x,point_y), axis=1)
        return point_xy
    def contour(point_xy, a = int(1)):
        import alphashape
        alpha = a
        alpha_shape = alphashape.alphashape(point_xy, alpha)
        x, y = alpha_shape.exterior.coords.xy
        x = np.array(x)
        y = np.array(y)
        return x, y
    def distance(x1,y1,x2,y2):
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)
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
            #         plt.plot(line_x, line_y_robust, '-b', label='Robust line model')
            line_twopoint = (line_x, line_y_robust)
        else:
            #         plt.plot(line_x_y, line_y_robust_y, '-b', label='Robust line model')
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
        if x_min -100 <= x <= x_max+ 100 and y_min - 100 <= y <= y_max + 100:
            return x, y
        else:
            return -12345, -12345
    def coord_sort(x):
        x = np.array(x)
        k = x[:,0]
        s = k.argsort()
        centers_sorted = x[s]
        for i in range(len(centers_sorted)//2):
            b = centers_sorted[2*i:2*(i+1),:]
            k = b[:,1]
            s = k.argsort()
            centers_sorted[2*i:2*(i+1), :] = b[s]
        return centers_sorted
    def img_to_coord(img):
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.flip(img, 0)
        return img
    def coord_to_img(coord):
        coord = cv2.flip(coord, 0)
        coord = cv2.rotate(coord, cv2.ROTATE_90_CLOCKWISE)
        return coord
    def homography(img, points, width = 300, height = 600):
        pts1 = np.float32(coord_sort(points))
        pts2 = np.float32([[0, 0], [width, 0],[0,height], [width, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img = img_to_coord(img)
        img = cv2.warpPerspective(img, matrix, (width, height))
        return img
    def cluster(point_data):
        from sklearn.cluster import KMeans
        k = 6
        model = KMeans(n_clusters=k)
        model.fit(point_data)
        predict = model.predict(point_data)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(set(predict))))
        k_means_labels = model.labels_
        k_means_cluster_centers = model.cluster_centers_
        fig = plt.figure(figsize=(6, 4))
        # plot 생성
        ax = fig.add_subplot(1, 1, 1)
        for k, col in zip(range(6), colors):
            my_members = (k_means_labels == k)

            # 중심 정의
            cluster_center = k_means_cluster_centers[k]
            #
            # # 중심 그리기
    #         ax.plot(point_data[my_members], point_data[my_members], 'w', markerfacecolor=col, marker='.')
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
        centers = np.array(k_means_cluster_centers, dtype=int)
        return centers
    def draw_dab_distances(centers):
        from shapely.geometry import Point
        centers_x = centers[:,0]
        centers_y = centers[:,1]
        distances_vert = []
        distances_horiz = []
        for k in range(0, 6 ,2):
            distances_vert.append(round(Point(centers[k]).distance(Point(centers[k+1]))*2,2))
            plt.plot((centers_x[k], centers_x[k+1]),(centers_y[k],centers_y[k+1]), marker="o",linestyle=":")
            plt.text((centers_x[k]+centers_x[k+1])/2 ,(centers_y[k]+centers_y[k+1])/2, distances_vert[-1], color="white")
        for k in range(0, 4):
            distances_horiz.append(round(Point(centers[k]).distance(Point(centers[k+2]))*2,2))
            plt.plot((centers_x[k], centers_x[k+2]),(centers_y[k],centers_y[k+2]), marker="s",linestyle="--")
            plt.text((centers_x[k]+centers_x[k+2])/2 ,(centers_y[k]+centers_y[k+2])/2, distances_horiz[-1],color="white")
    def img_trans(img):
        img_trans = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_trans = cv2.inRange(img_trans, (0, 0, 0), (200, 15, 200))
        adhesive = np.where(img_trans == 0)
        rest = np.where(img_trans == 255)
        return adhesive, rest, img_trans
    ransac_line = []
    intersection_points = []
    src = sizing(src, coord)
    point_xy = seek(coord)
    x_data, y_data = contour(point_xy)
    x_tmp = x_data.copy()
    y_tmp = y_data.copy()
    while True:
        inliers, outliers, line_twopoint = plot_ransac_revised(x_tmp, y_tmp)

        if x_tmp[inliers].shape[0] >= 2:
            # inliers, two points for line 기록 저장
            ransac_line.append((x_tmp[inliers], y_tmp[inliers], line_twopoint))

        # 나머지 점들 (outliers)
        x_tmp = x_tmp[outliers]
        y_tmp = y_tmp[outliers]

        if x_tmp.shape[0] <= 2 or len(ransac_line) == 4:
            break
    x_min, x_max, y_min, y_max = x_data.min(), x_data.max(), y_data.min(), y_data.max()
    for i in range(len(ransac_line)):
        for j in range(i+1, len(ransac_line)):
            (x1, x2), (y1, y2)= ransac_line[i][2]
            (x3, x4), (y3, y4)= ransac_line[j][2]
            x, y = line_intersection([[x1, y1], [x2, y2]], [[x3, y3], [x4, y4]], x_min, x_max, y_min, y_max)
            if x != -12345 or y != -12345:
                intersection_points.append(np.array((x,y)))
    imgOutput = homography(src, intersection_points)
    adhesive, rest, img_trans = img_trans(imgOutput)
    ratio = round(np.sum(adhesive) / (np.sum(rest) + np.sum(adhesive)) * 100, 2)
    print("접착 면적 비율:", round(np.sum(adhesive) / (np.sum(rest) + np.sum(adhesive)) * 100, 2))
    coordOutput = homography(coord, intersection_points)
    k = seek(coordOutput, b=0)
    centers = cluster(k)
    centers = coord_sort(centers)
    draw_dab_distances(centers)
    plt.text(50,50,"ratio : {}%".format(ratio), color="white", fontweight="semibold")
    plt.imshow(img_to_coord(imgOutput))
    plt.show()
src = "./segmented_figure_final.jpg"
coord = np.loadtxt("./output_6.csv", delimiter=",")
EIFS_automation(src, coord)