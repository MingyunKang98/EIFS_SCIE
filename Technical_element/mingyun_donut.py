import os
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np

def distance(point1, point2):
    return np.linalg.norm(point1 - point2)
def parse_polygon_annotation(annotation_line, img_width, img_height):
    parts = list(map(float, annotation_line.split()))
    class_id = int(parts[0])
    points = [(int(x * img_width), int(y * img_height)) for x, y in zip(parts[1::2], parts[2::2])]
    points = np.array(points)
    return class_id, points

def find_closest_points(contour_outer, contour_inner):
    min_dist = float('inf')
    closest_points = None

    for point_outer in contour_outer:
        for point_inner in contour_inner:
            dist = np.linalg.norm(point_outer - point_inner)  # 두 점 간의 거리 계산
            if dist < min_dist:
                min_dist = dist
                closest_points = (point_outer, point_inner)

    return closest_points


# 바깥 contour와 안쪽 contour의 점들을 받아서 하나의 폴리곤으로 만들어주는 함수
def connect_contours(contour_outer, contour_inner):
    closest_points = find_closest_points(contour_outer, contour_inner)
    if closest_points is None:
        return None
    polygon = np.concatenate((np.flip(contour_outer, axis=0), contour_inner))  # 바깥 contour는 반시계방향으로 배열하고, 안쪽 contour는 시계방향으로 배열한 후 연결

    return polygon



if __name__ == "__main__":
    image_path = "../SCIE/EIFS_Donut_Only_ribbon/train/1.jpg"
    annotation_path = "../SCIE/EIFS_Donut_Only_ribbon/train/annotations_YOLO_only_ribbon/1.txt"
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]

    with open(annotation_path, 'r', encoding='utf-8') as file:
        line = file.readlines()

    class_id_outer, polygon_points_outer = parse_polygon_annotation(line[0], img_width, img_height)
    class_id_inner, polygon_points_inner = parse_polygon_annotation(line[1], img_width, img_height)

    polygon = connect_contours(polygon_points_outer, polygon_points_inner)

    # 폴리곤 시각화
    plt.plot(polygon[:, 0], polygon[:, 1], 'b-')  # 폴리곤을 파란색 선으로 표시
    plt.imshow(image)
    plt.show()