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

# 사용 예시
input_file = "./SCIE/EIFS_Doughnut_2.v1i.coco-segmentation/train/_annotations.coco.json"
output_file = "./output.json"
format_and_save_json(input_file, output_file)