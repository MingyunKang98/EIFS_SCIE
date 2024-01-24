import numpy as np
import pandas as pd
import cv2
import math
import copy


def RX(rx):
    return np.array([[1., 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]], dtype=np.float32)


def RY(ry):
    return np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]], dtype=np.float32)


def RZ(rz):
    return np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]], dtype=np.float32)


def main():
    input = "./SCI_datasets/20230509_102031.jpg"
    f, cx, cy, L = 810.5, 480, 270, 3.31
    cam_ori = np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)])    # np.array([np.deg2rad(-18.7), np.deg2rad(-8.2), np.deg2rad(2.0)])
    grid_x, grid_z = (-4, 3), (4, 10)    # (-2, 3), (5, 35)

    # Load an images
    image = cv2.imread(input)
    # if image == None: return -1

    # Configure mouse callback
    # drag = MouseDrag()
    cv2.namedWindow("3DV Tutorial: Pbject Localization and Measurement")
    # cv2.setMouseCallback("3DV Tutorial: Pbject Localization and Measurement", MouseEventHandler, drag)  # ?

    # Draw grids on the ground
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    Rc = RZ(cam_ori[2]) @ RY(cam_ori[1]) @ RX(cam_ori[0])
    R = Rc.T
    tc = np.array([[0, -L, 0]], dtype=np.float32)
    t = -Rc.T @ tc.T
    for z in range(grid_z[0], grid_z[1], 1):
        a, b = np.array([[grid_x[0], 0, z]], dtype=np.float32), np.array([[grid_x[1], 0, z]], dtype=np.float32)
        p = K @ (R @ a.T + t)
        q = K @ (R @ b.T + t)
        image = cv2.line(image, (int(p[0] / p[2]), int(p[1] / p[2])), (int(q[0] / q[2]), int(q[1] / q[2])),
                         (64, 128, 64), 1)

    for x in range(grid_x[0], grid_x[1]):
        a, b = np.array([[x, 0, grid_z[0]]], dtype=np.float32), np.array([[x, 0, grid_z[1]]], dtype=np.float32)
        p = K @ (R @ a.T + t)
        q = K @ (R @ b.T + t)
        image = cv2.line(image, (int(p[0] / p[2]), int(p[1] / p[2])), (int(q[0] / q[2]), int(q[1] / q[2])),
                         (64, 128, 64), 1)


    image_copy = copy.deepcopy(image)
############################################################################################################


    file_path = "./SCI_datasets/20230509_102031.txt"
    with open(file_path) as fp:
        lines = fp.readlines()

    coord = []
    height = image.shape[0]
    width = image.shape[1]
    for line in lines:
        line_split = line.split()
        line_split_float = list(map(float, line_split))
        np_float = np.array(line_split_float[1:])
        np_float = np_float.reshape(-1, 2)
        np_float[:,0] = np_float[:,0]*widthㅂㅂㅂㅂㅂㅂㅂㅂㅂㅂㅂㅂ
        np_float[:,1] = np_float[:,1]*height
        np_float = np.asarray(np_float, dtype = int)

        coord.append(np_float)

    max_y = []
    for support in coord:
        tmp_y = np.max(support, axis=0)
        max_y.append(tmp_y)

    median_x = []
    real_cen_point = []
    for support, xy in zip(coord, max_y):
        df = pd.DataFrame(support, columns=['x', 'y'])
        x_group = df[df['y'] > xy[1]-5]   # y값의 최대 > 최대값에서 -5 / y값들의 범위지정
        min_x = x_group.min(axis=0)  # x만 쓸수있음. (동일 좌표의 x,y가 아님)
        max_x = x_group.max(axis=0)  # x만 쓸수있음. (동일 좌표의 x,y가 아님)

        # x의 최솟값
        min_x_idx = np.where(x_group==min_x[0])
        idx = min_x_idx[0][0]
        min_xy = x_group.iloc[idx]

        # x의 최대값
        max_x_idx = np.where(x_group==max_x[0])
        idx = max_x_idx[0][0]
        max_xy = x_group.iloc[idx]

        # print(min_xy)
        # print(max_xy)

        cen_point = (int((min_xy[0]+max_xy[0])/2), int((min_xy[1]+max_xy[1])/2))
        real_cen_point.append(cen_point)
        median_x.append(x_group.median(axis=0))



    print("cen_point: ", real_cen_point)


        ############################################################################################################

    two_real_dot = []
    for i in range(len(real_cen_point)):
        A_XY = real_cen_point[i]
        a, b = np.array([[A_XY[0] - cx, A_XY[1] - cy, f]], dtype=np.float32), np.array(
            [[A_XY[0] - cx, A_XY[1] - cy, f]], dtype=np.float32)
        c, h = R.T @ a.T, R.T @ b.T
        # if c[1] # 정수비교가 필요하지만, 파이썬에서는 어떻게 하는지 모르므로 패스
        X = c[0] / c[2] * L
        Z = c[2] / c[1] * L
        H = (c[1] / c[2] - h[1] / h[2]) * Z

        A_info = X[0], Z[0]

        # A_info = f"X: {X[0]:.3f}, Z: {Z[0]:.3f}, H: {H[0]:.3f}"
        two_real_dot.append(A_info)

        for j in range(i+1, len(real_cen_point)):
            B_XY = real_cen_point[j]
            a, b = np.array([[B_XY[0] - cx, B_XY[1] - cy, f]], dtype=np.float32), np.array(
                [[B_XY[0] - cx, B_XY[1] - cy, f]], dtype=np.float32)
            c, h = R.T @ a.T, R.T @ b.T
            # if c[1] # 정수비교가 필요하지만, 파이썬에서는 어떻게 하는지 모르므로 패스
            X = c[0] / c[2] * L
            Z = c[2] / c[1] * L
            H = (c[1] / c[2] - h[1] / h[2]) * Z

            B_info = X[0], Z[0]

            # B_info = f"X: {X[0]:.3f}, Z: {Z[0]:.3f}, H: {H[0]:.3f}"
            two_real_dot.append(B_info)

            # 거리계산

            distance = ((two_real_dot[0][0] - two_real_dot[1][0])**2 + (two_real_dot[0][1] - two_real_dot[1][1])**2)**(1/2)
            print(distance)

            # Draw head/contact points and location/height
            info = f"X: {X[0]:.3f}, Z: {Z[0]:.3f}, H: {H[0]:.3f}"
            image_copy = cv2.line(image_copy, A_XY, B_XY, (0, 0, 255), 2)
            image_copy = cv2.circle(image_copy, A_XY, 4, (255, 0, 0), -1)
            image_copy = cv2.circle(image_copy, B_XY, 4, (0, 255, 0), -1)
            image_copy = cv2.putText(image_copy, info, (A_XY[0] - 20, A_XY[1] + 20), cv2.FONT_HERSHEY_PLAIN,
                                     1, (0, 255, 0))

            # cv2.line(image_copy, (10, 640), (158, 632), (0, 255, 255), 5)
            cv2.imshow("3DV Tutorial: Pbject Localization and Measurement", image_copy)
            cv2.waitKey(0)


if __name__ == "__main__":
    main()



cen_point:  (0.357792, 0.9085935)
cen_point:  (0.042819315000000004, 0.5367185)
cen_point:  (0.5399594999999999, 0.18906299999999998)
cen_point:  (0.8655094999999999, 0.3234375)
cen_point:  (0.1650475, 0.42343800000000004)
cen_point:  (0.6809915, 0.104687)