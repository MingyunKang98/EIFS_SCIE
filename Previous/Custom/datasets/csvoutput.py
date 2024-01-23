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