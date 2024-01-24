import numpy as np
import cv2
import math
import copy


def RX(rx):
    return np.array([[1., 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]], dtype=np.float32)


def RY(ry):
    return np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]], dtype=np.float32)


def RZ(rz):
    return np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]], dtype=np.float32)


class MouseDrag():
    def __init__(self):
        self._dragged = False
        self.start, self.end = (1, 1), (1, 1)


def MouseEventHandler(event, x, y, flags, params):
    if params == None: return
    if event == cv2.EVENT_LBUTTONDOWN:
        params._dragged = True
        params.start = (x, y)
        params.end = (0, 0)
    elif event == cv2.EVENT_MOUSEMOVE:
        if params._dragged:
            params.end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if params._dragged:
            params._dragged = False
            params.end = (x, y)


def main():
    input = "./daejeon_station.png"
    f, cx, cy, L = 810.5, 480, 270, L   # L = 3.31
    cam_ori = np.array([np.deg2rad(-18.7), np.deg2rad(-8.2), np.deg2rad(2.0)])
    grid_x, grid_z = (-2, 3), (5, 35)

    # Load an image
    image = cv2.imread(input)
    # if image == None: return -1

    # Configure mouse callback
    drag = MouseDrag()
    cv2.namedWindow("3DV Tutorial: Pbject Localization and Measurement")
    cv2.setMouseCallback("3DV Tutorial: Pbject Localization and Measurement", MouseEventHandler, drag)

    # Draw grids on the ground
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)   # 카메라 고유 매트릭스를 사용하여 2D 이미지 평면에 3D 포인트를 투영하는 것과 관련된 지면에 그리드를 그리는 코드
    Rc = RZ(cam_ori[2]) @ RY(cam_ori[1]) @ RX(cam_ori[0])
    R = Rc.T
    tc = np.array([[0, -L, 0]], dtype=np.float32)       # 카메라 좌표계에서 카메라에 대한 이동 벡터를 나타내는 변수입니다. 카메라 외부 행렬 t의 계산에 사용됩니다.
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




    # while True:
    #     image_copy = copy.deepcopy(image)
    #     if drag.end[0] > 0 and drag.end[1] > 0:
    #         # Calculate object location and height
    #         a, b = np.array([[drag.start[0] - cx, drag.start[1] - cy, f]], dtype=np.float32), np.array(
    #             [[drag.end[0] - cx, drag.end[1] - cy, f]], dtype=np.float32)
    #         c, h = R.T @ a.T, R.T @ b.T
    #         # if c[1] # 정수비교가 필요하지만, 파이썬에서는 어떻게 하는지 모르므로 패스
    #         X = c[0] / c[2] * H / (1-(h[1] / h[2] * c[2] / c[1]))      # Y
    #         Z = c[2] / c[1] * H / (1-(h[1] / h[2] * c[2] / c[1]))      # X
    #         L = np.sqrt(X ** 2 + Z ** 2)  # L
    #         image_copy = cv2.line(image_copy, drag.start, drag.end, (0, 255, 255), 2)
    #         cv2.imshow("3DV Tutorial: Pbject Localization and Measurement", image_copy)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #     cv2.destroyAllWindows()

    while True:
        image_copy = copy.deepcopy(image)
        if drag.end[0] > 0 and drag.end[1] > 0:
            # Calculate object location and height
            a, b = np.array([[drag.start[0] - cx, drag.start[1] - cy, f]], dtype=np.float32), np.array(
                [[drag.end[0] - cx, drag.end[1] - cy, f]], dtype=np.float32)
            c, h = R.T @ a.T, R.T @ b.T
            X = c[0] / c[2] * L / (1 - h[1] / h[2] * c[2] / c[1])  # Y
            Z = c[2] / c[1] * X  # X
            L = np.sqrt(X ** 2 + Z ** 2)  # L
            image_copy = cv2.line(image_copy, drag.start, drag.end, (0, 255, 255), 2)
            cv2.imshow("3DV Tutorial: Pbject Localization and Measurement", image_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()






            # H = 1.74   # H
            # L =  H / (1-(h[1] / h[2] * c[2] / c[1]))






            # Draw head/contact points and location/height
            info = f"X: {X[0]:.3f}, Z: {Z[0]:.3f}, H: {H[0]:.3f}"
            image_copy = cv2.line(image_copy, drag.start, drag.end, (0, 0, 255), 2)
            image_copy = cv2.circle(image_copy, drag.end, 4, (255, 0, 0), -1)
            image_copy = cv2.circle(image_copy, drag.start, 4, (0, 255, 0), -1)
            image_copy = cv2.putText(image_copy, info, (drag.start[0] - 20, drag.start[1] + 20), cv2.FONT_HERSHEY_PLAIN,
                                     1, (0, 255, 0))

        cv2.imshow("3DV Tutorial: Pbject Localization and Measurement", image_copy)
        if cv2.waitKey(1) == ord('q'): break


if __name__ == "__main__":
    main()