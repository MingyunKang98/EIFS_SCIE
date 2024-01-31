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
