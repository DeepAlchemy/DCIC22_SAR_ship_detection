def generate_bg_label(label_path):
    import os
    files = os.listdir(label_path)
    for file_name in files:
        file_path = "{}/{}".format(label_path, file_name)
        file = open(file_path)
        print(file_path)
        bg_label_list = []
        for line in file:
            data_split = line.replace("\n", "").split()
            cx = float(data_split[1])
            cy = float(data_split[2])
            w = float(data_split[3])
            h = float(data_split[4])
            new_cx = cx + w / 2
            new_cy = cy + h / 2
            if new_cx < 1 and new_cy < 1:
                bg_label = " ".join(["1", str(new_cx), str(new_cy), str(w), str(h), "\n"])
                bg_label_list.append(bg_label)

        if bg_label_list:
            file = open(file_path, 'a')
            for label in bg_label_list:
                print(label)
                file.write(label)
            file.close()


if __name__ == '__main__':
    path = '/Users/kevin/Downloads/DCIC2022_dataset/labels/val'
    generate_bg_label(path)
