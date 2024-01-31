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
