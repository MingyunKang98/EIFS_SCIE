# import json
# import os
#
# def convert_coco_to_yolo_segmentation(coco_file, output_dir):
#     # JSON 파일 읽기
#     with open(coco_file, 'r') as file:
#         coco = json.load(file)
#
#     annotations_per_image = {}
#     for ann in coco['annotations']:
#         image_id = ann['image_id']
#         category_id = ann['category_id'] - 1  # YOLO 클래스는 0부터 시작
#         segmentation = ann['segmentation'][0]  # 첫 번째 세그멘테이션 다각형 사용
#
#         # 이미지의 너비와 높이로 나누어 정규화
#         img_data = next((item for item in coco['images'] if item["id"] == image_id), None)
#         img_width, img_height = img_data['width'], img_data['height']
#         normalized_segmentation = [str(category_id)] + [str(point / img_width if i % 2 == 0 else point / img_height) for i, point in enumerate(segmentation)]
#
#         # 어노테이션 저장
#         annotation = " ".join(normalized_segmentation)
#         if image_id not in annotations_per_image:
#             annotations_per_image[image_id] = []
#         annotations_per_image[image_id].append(annotation)
#
#     # 파일로 저장
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     for image_id, annotations in annotations_per_image.items():
#         file_name = f"{image_id}.txt"
#         with open(os.path.join(output_dir, file_name), 'w') as file:
#             file.write("\n".join(annotations))
#
# # 사용 예시
# coco_annotation_file = "./train/_annotations.coco.json"
# output_directory = "./train/annoation_YOLO.txt"
# convert_coco_to_yolo_segmentation(coco_annotation_file, output_directory)

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
coco_annotation_file = "./train/_annotations.coco.json"
output_directory = "./train/annoation_YOLO.txt"
convert_coco_to_yolo_and_match_filenames(coco_annotation_file, output_directory)

