import os
import glob

import cv2
import numpy as np


def distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def find_closest_points(contour_outer, contour_inner):
    min_dist = np.inf
    closest_points = None

    for point_outer in contour_outer:
        for point_inner in contour_inner:
            point_outer_arr = np.array(point_outer)
            point_inner_arr = np.array(point_inner)
            dist = np.linalg.norm(point_outer_arr - point_inner_arr)
            if dist < min_dist:
                min_dist = dist
                closest_points = (point_outer_arr, point_inner_arr)

    return closest_points


# 바깥 contour와 안쪽 contour의 점들을 받아서 하나의 폴리곤으로 만들어주는 함수
def connect_contours(contour_outer, contour_inner):
    closest_points = find_closest_points(contour_outer, contour_inner)
    if closest_points is None:
        return None
    polygon = np.concatenate(
        (np.flip(contour_outer, axis=0), contour_inner))  # 바깥 contour는 반시계방향으로 배열하고, 안쪽 contour는 시계방향으로 배열한 후 연결

    return polygon


def parse_class_annotation(line):
    parts = list(map(float, line.split()))
    class_id = int(parts[0])
    points = [(x, y) for x, y in zip(parts[1::2], parts[2::2])]
    poinst = np.array(points)
    return class_id, points


# 이미지 경로와 어노테이션 파일 경로
annotation_path = "./train/labels/IMG_4245_JPG.rf.63460a44ffb7591e4ab7aaa7483d11e6.txt"

# 어노테이션 파일 경로를 이용하여 모든 txt 파일의 경로를 가져옴
txt_files = glob.glob(annotation_path)

# 각 파일에 대해 작업 수행
for file_path in txt_files:
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    class_id_outer, polygon_points_outer = parse_class_annotation(lines[1])
    class_id_inner, polygon_points_inner = parse_class_annotation(lines[0])
    polygon = connect_contours(polygon_points_outer, polygon_points_inner)

    donut_ribbon = np.append(class_id_outer, polygon)
    donut_ribbon = donut_ribbon.tolist()
    donut_ribbon[0] = 1
    dab = np.array(lines[2:])

    dab_list = []
    for i in range(2, 10):
        dab, points = parse_class_annotation(lines[i])
        dab_combined = np.append(dab, points)
        dab_list.append(dab_combined.tolist())

    merged = [donut_ribbon] + dab_list

    for i in range(1, 9):
        merged[i][0] = 0

    with open(file_path, 'w') as file:
        for row in merged:
            row_str = ' '.join(map(str, row))  # 각 행의 요소를 문자열로 변환하고 탭으로 구분
            file.write(row_str + '\n')  # 행을 파일에 쓰고 줄 바꿈 추가
