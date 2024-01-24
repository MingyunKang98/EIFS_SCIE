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

