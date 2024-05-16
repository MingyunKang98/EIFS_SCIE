# SCIE 를 위한 여정

### Previous Code review

1. YOLO를 통해 추출된 각 class별 output을 ```csvoutput.py```를 통해 csv 로 변환

csvoutput.py

```
dir = "1.txt"
for k in range(12):
    coord = [[], [], [], [], [], [], []]
    idx = -1
    with open(dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split()
            if len(tmp) > 100:
                tmp = list(map(float, tmp))
                coord[idx].append(tmp)
            else:
                idx += 1
    import csv
    for i in range(len(coord)):
        with open("output_{}.csv".format(i), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(coord[i])
    total_class = 2
    coord_idx = {}
    for idx in range(total_class):
        coord_idx[idx]=[]
  ```

2. ```KIC_Final.py```를 통해 자동감리 모델을 구현
  이때 각 output_*에 따라 모델이 구현될수도 있고 안될 수도 있음

  에러 원인 1. Ransac시 기울기가 무한대로 발산할 수 있음 2. output_*.csv 이 dab을 제외한 insulation이 아닐 수 있음


---

## 연구일지

### 240131
1. Roboflow json annotation -> _annotations.coco.json 파일 생성


2. JsonFormatter.py -> 정렬된 json 파일 생성

```
import json

def format_and_save_json(input_file, output_file):
    # JSON 파일 읽기
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 데이터를 예쁘게 정렬
    formatted_data = json.dumps(data, indent=4, ensure_ascii=False)

    # 새 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(formatted_data)
    print(f"File '{input_file}' has been formatted and saved as '{output_file}'.")


input_file = "./directory"
output_file = "./directory"
format_and_save_json(input_file, output_file)
```


3. json_txt_converter.py -> json annotation을 YOLO.txt파일로 변환 파일 생성

```
import json
import os

def convert_coco_to_yolo_and_match_filenames(coco_file, output_dir, img_extension=".jpg"):
    # JSON 파일 읽기
    with open(coco_file, 'r') as file:
        coco = json.load(file)

    annotations_per_image = {}
    for ann in coco['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id'] - 1  # YOLO 클래스는 0부터 시작
        segmentation = ann['segmentation'][0]  # 첫 번째 세그멘테이션 다각형 사용

        # 이미지의 너비와 높이로 나누어 정규화
        img_data = next((item for item in coco['images'] if item["id"] == image_id), None)
        img_width, img_height = img_data['width'], img_data['height']
        img_file_name = os.path.splitext(img_data['file_name'])[0]  # 확장자 제거
        normalized_segmentation = [str(category_id)] + [str(point / img_width if i % 2 == 0 else point / img_height) for i, point in enumerate(segmentation)]

        # 어노테이션 저장
        annotation = " ".join(normalized_segmentation)
        if img_file_name not in annotations_per_image:
            annotations_per_image[img_file_name] = []
        annotations_per_image[img_file_name].append(annotation)

    # 파일로 저장
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_file_name, annotations in annotations_per_image.items():
        file_name = f"{img_file_name}.txt"
        with open(os.path.join(output_dir, file_name), 'w') as file:
            file.write("\n".join(annotations))

# 사용 예시
coco_annotation_file = "./directory"
output_directory = "./directory"
convert_coco_to_yolo_and_match_filenames(coco_annotation_file, output_directory)
```

## 합쳤음!! -> NEW YOLO CONVERTER (json format + YOLO Converter)

```
import json
import os

def format_and_save_json(input_file):
    # JSON 파일 읽기
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 데이터를 예쁘게 정렬
    formatted_data = json.dumps(data, indent=4, ensure_ascii=False)
    return formatted_data

def convert_coco_to_yolo_and_match_filenames(coco_json_str, output_dir, img_extension=".jpg"):
    # 문자열에서 JSON 객체로 변환
    coco = json.loads(coco_json_str)

    annotations_per_image = {}
    for ann in coco['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id'] - 1  # YOLO 클래스는 0부터 시작
        segmentation = ann['segmentation'][0]  # 첫 번째 세그멘테이션 다각형 사용

        # 이미지의 너비와 높이로 나누어 정규화
        img_data = next((item for item in coco['images'] if item["id"] == image_id), None)
        img_width, img_height = img_data['width'], img_data['height']
        img_file_name = os.path.splitext(img_data['file_name'])[0]  # 확장자 제거
        normalized_segmentation = [str(category_id)] + [str(point / img_width if i % 2 == 0 else point / img_height) for i, point in enumerate(segmentation)]

        # 어노테이션 저장
        annotation = " ".join(normalized_segmentation)
        if img_file_name not in annotations_per_image:
            annotations_per_image[img_file_name] = []
        annotations_per_image[img_file_name].append(annotation)

    # 파일로 저장
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_file_name, annotations in annotations_per_image.items():
        file_name = f"{img_file_name}{img_extension}.txt"
        with open(os.path.join(output_dir, file_name), 'w') as file:
            file.write("\n".join(annotations))

if __name__ == "__main__":
    # 사용 예시
    input_file = "train/_annotations.coco.json"
    formatted_json = format_and_save_json(input_file)
    output_directory = "./train/annotations_YOLO"
    convert_coco_to_yolo_and_match_filenames(formatted_json, output_directory)
```

# 240401 mingyundonut

```
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
```
# 240409 Donut_Annotation 추가 하였음

```commandline
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


annotation_path = "./train/labels/IMG_4235_JPG.rf.c374d85c12b44480a90c6f1eb67932e4.txt"

with open(annotation_path, 'r', encoding='utf-8') as file:
    line = file.readlines()

class_id_outer, polygon_points_outer = parse_class_annotation(line[1])
class_id_inner, polygon_points_inner = parse_class_annotation(line[0])
polygon = connect_contours(polygon_points_outer, polygon_points_inner)

donut_ribbon = np.append(class_id_outer,polygon)
donut_ribbon = donut_ribbon.tolist()
donut_ribbon[0] = 1
dab = np.array(line[2:])

dab_list = []
for i in range(2, 10):
    dab, points = parse_class_annotation(line[i])
    dab_combined = np.append(dab, points)
    dab_list.append(dab_combined.tolist())
# donut_ribbon = [donut_ribbon]

merged = [donut_ribbon]+dab_list

for i in range(1, 9):
    merged[i][0] = 0

file_path = annotation_path

with open(file_path, 'w') as file:
    for row in merged:
        row_str = ' '.join(map(str, row))  # 각 행의 요소를 문자열로 변환하고 탭으로 구분
        file.write(row_str + '\n')  # 행을 파일에 쓰고 줄 바꿈 추가

```



