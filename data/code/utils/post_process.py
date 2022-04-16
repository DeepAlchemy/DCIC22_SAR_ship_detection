import json
import os
import csv

#   YOLO:  (xmin, ymin, xmax, ymax)
#   COCO:  (x1, y1, width, height)
class ResultUtil:

    @classmethod
    def labels_process(cls, sample_path, label_path):
        """
        标签处理器
        :param sample_path: 样本目录路径
        :param label_path:  标签目录路径
        :return:
        """
        sample_list = {}
        with open(sample_path, newline="") as sample_file:
            samples = csv.reader(sample_file)
            for item in samples:
                sample_list[item[0]] = ""
        files = os.listdir(label_path)
        for file_name in files:
            file_id = str(int(file_name.replace(".txt", "")))
            file_path = "{}/{}".format(label_path, file_name)
            file = open(file_path)
            data = ""
            for line in file:
                if data == "":
                    # 去掉最后的阈值
                    data_split = line.replace("0 ", "").replace("\n", "").split()
                    data += " ".join(data_split[0: 4])
                else:
                    data_split = line.replace("0 ", "").replace("\n", "").split()
                    data += ";{}".format(" ".join(data_split[0: 4]))
            sample_list[file_id] = data
        result_list = []
        for item in sample_list:
            result_list.append([item, sample_list[item]])
        with open("../../prediction_result/result.csv", "w") as submission:
            writer = csv.writer(submission)
            writer.writerows(result_list)

    @classmethod
    def submit_format_converter(cls, test_file_path, test_result_path):
        """
        提交格式转换器，由COCO格式转为YOLO格式
        :param test_file_path:      测试文件路径
        :param test_result_path:    测试结果文件路径
        """
        results = []
        converted_results = []

        with open(test_file_path, "r") as json_str:
            test_files = json.loads(json_str.read())["images"]
        with open(test_result_path, "r") as json_str:
            bbox_results = json.loads(json_str.read())

        for test_file in test_files:
            file_id = int(test_file["file_name"].replace(".jpg", ""))
            item = {
                "id": file_id,
                "location": []
            }
            for bbox_result in bbox_results:
                if bbox_result["image_id"] == test_file["id"] and bbox_result["score"] >= 0.7 and bbox_result['category_id'] == 1:
                # if bbox_result["image_id"] == test_file["id"]:
                    item["location"].append(ResultUtil.coco_2_yolo(bbox_result["bbox"]))
            results.append(item)

        for result in results:
            new_result = ""
            if len(result["location"]) != 0:
                items = []
                for location in result["location"]:
                    item = [str(i) for i in location]
                    items.append(item)
                if len(items) == 1:
                    new_result = " ".join(items[0])
                else:
                    for item in items:
                        new_result += " ".join(item) if new_result == "" else ";{}".format(" ".join(item))
            converted_results.append([result["id"], new_result])

        with open("../../prediction_result/result.csv", "w") as submission:
            writer = csv.writer(submission)
            writer.writerows(sorted(converted_results, key=lambda x: x[0]))

    @classmethod
    def coco_2_yolo(cls, box, size=None):
        '''
        COCO格式转YOLO格式。测试输出的COCO格式，比赛要求提交YOLO格式
        size: 图片的宽和高(w,h), 默认值为[256, 256]
        box格式: x,y,w,h
        返回值：x_center/image_width y_center/image_height width/image_width height/image_height
        '''

        if size is None:
            size = [256, 256]
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = box[0] + box[2] / 2.0
        y = box[1] + box[3] / 2.0
        w = box[2]
        h = box[3]

        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return [round(x, 8), round(y, 8), round(w, 8), round(h, 8)]

    @classmethod
    def yolofCoco2Yolo(self, sample_path, label_path):
        """
        标签处理器
        :param sample_path: 样本目录路径
        :param label_path:  标签目录路径
        :return:
        """
        sample_list = {}
        with open(sample_path, newline="") as sample_file:
            samples = csv.reader(sample_file)
            for item in samples:
                sample_list[item[0]] = ""
        files = os.listdir(label_path)
        for file_name in files:
            file_id = str(int(file_name.replace(".txt", "")))
            file_path = "{}/{}".format(label_path, file_name)
            file = open(file_path)
            data = ""
            for line in file:
                if data == "":
                    # 去掉最后的阈值
                    data_split = line.replace("\n", "").split()
                    coor = data_split[0: 4]
                    float_list = list(map(float, coor))
                    data += " ".join(ResultUtil.xyxy2cxcywh(float_list))
                else:
                    data_split = line.replace("\n", "").split()
                    coor = data_split[0: 4]
                    float_list = list(map(float, coor))
                    data += ";{}".format(" ".join(ResultUtil.xyxy2cxcywh(float_list)))
            sample_list[file_id] = data
        result_list = []
        for item in sample_list:
            result_list.append([item, sample_list[item]])
        with open("../../prediction_result/result.csv", "w") as submission:
            writer = csv.writer(submission)
            writer.writerows(result_list)

    @classmethod
    def wbf2submission(self, sample_path, label_path):
        """
        标签处理器
        :param sample_path: 样本目录路径
        :param label_path:  标签目录路径
        :return:
        """
        sample_list = {}
        with open(sample_path, newline="") as sample_file:
            samples = csv.reader(sample_file)
            for item in samples:
                sample_list[item[0]] = ""
        files = os.listdir(label_path)
        for file_name in files:
            file_id = str(int(file_name.replace(".txt", "")))
            file_path = "{}/{}".format(label_path, file_name)
            file = open(file_path)
            data = ""
            for line in file:
                if data == "":
                    data_split = line.replace("\n", "").split(',')
                    coor = data_split[0: 4]
                    float_list = list(map(float, coor))
                    data += " ".join(ResultUtil.wbfxyxy2cxcywh(float_list))
                else:
                    data_split = line.replace("\n", "").split(',')
                    coor = data_split[0: 4]
                    float_list = list(map(float, coor))
                    data += ";{}".format(" ".join(ResultUtil.wbfxyxy2cxcywh(float_list)))
            sample_list[file_id] = data
        result_list = []
        for item in sample_list:
            result_list.append([item, sample_list[item]])
        # current_path = os.getcwd()
        # result_path = os.path.join(current_path, "../../prediction_result/result.csv")
        with open("/data/prediction_result/result.csv", "w") as submission:
        # with open("../../prediction_result/result.csv", "w") as submission:
            writer = csv.writer(submission)
            writer.writerows(result_list)


    @classmethod
    def yolofCoco2YoloWithClsLabel(self, sample_path, label_path):
        """
        标签处理器
        :param sample_path: 样本目录路径
        :param label_path:  标签目录路径
        :return:
        """
        sample_list = {}
        with open(sample_path, newline="") as sample_file:
            samples = csv.reader(sample_file)
            for item in samples:
                sample_list[item[0]] = ""
        files = os.listdir(label_path)
        for file_name in files:
            file_id = str(int(file_name.replace(".txt", "")))
            file_path = "{}/{}".format(label_path, file_name)
            file = open(file_path)
            data = ""
            for line in file:
                if data == "":
                    # 去掉最后的阈值
                    data_split = line.replace("\n", "").split()
                    cls = data_split[0]
                    if cls == '0':
                        coor = data_split[1: 5]
                        float_list = list(map(float, coor))
                        data += " ".join(ResultUtil.xyxy2cxcywh(float_list))
                else:
                    data_split = line.replace("\n", "").split()
                    cls = data_split[0]
                    if cls == '0':
                        coor = data_split[1: 5]
                        float_list = list(map(float, coor))
                        data += ";{}".format(" ".join(ResultUtil.xyxy2cxcywh(float_list)))
            sample_list[file_id] = data
        result_list = []
        for item in sample_list:
            result_list.append([item, sample_list[item]])
        with open("/data/prediction_result/result.csv", "w") as submission:
            writer = csv.writer(submission)
            writer.writerows(result_list)

    @classmethod
    def xyxy2cxcywh(self, box, size=None):
        '''
        左上角xy和右下角xy转成中间x和中间y以及长宽的相对比例
        size: 图片的宽和高(w,h), 默认值为[256, 256]
        box格式: x,y,x,y 左上角xy和右下角xy
        返回值：x_center/image_width y_center/image_height width/image_width height/image_height
        '''
        if size is None:
            size = [256, 256]
        dw = 1. / (size[0])
        dh = 1. / (size[1])

        w = box[2] - box[0]
        h = box[3] - box[1]
        cx = box[0] + w * 0.5
        cy = box[1] + h * 0.5

        x = cx * dw
        w = w * dw
        y = cy * dh
        h = h * dh

        float_list = [round(x, 8), round(y, 8), round(w, 8), round(h, 8)]
        ret_list = list(map(str, float_list))
        print(ret_list)
        return ret_list

    @classmethod
    def wbfxyxy2cxcywh(self, box, size=None):
        '''
        左上角xy和右下角xy转成中间x和中间y以及长宽的相对比例
        size: 图片的宽和高(w,h), 默认值为[256, 256]
        box格式: x,y,x,y 左上角xy和右下角xy
        返回值：x_center/image_width y_center/image_height width/image_width height/image_height
        '''
        if size is None:
            size = [256, 256]
        dw = 1. / (size[0])
        dh = 1. / (size[1])

        rescale_box = []
        for i in box:
            i *= 256 # rescale to 0-256
            rescale_box.append(i)

        w = rescale_box[2] - rescale_box[0]
        h = rescale_box[3] - rescale_box[1]
        cx = rescale_box[0] + w * 0.5
        cy = rescale_box[1] + h * 0.5

        x = cx * dw
        w = w * dw
        y = cy * dh
        h = h * dh

        float_list = [round(x, 8), round(y, 8), round(w, 8), round(h, 8)]
        ret_list = list(map(str, float_list))
        print(ret_list)
        return ret_list


if __name__ == '__main__':
    # YOLOv5 post process
    # sample_path = "/Users/kevin/Downloads/DCIC2022_dataset/submit_example.csv"
    # label_path = "/Users/kevin/Downloads/labels"
    # ResultUtil.labels_process(sample_path, label_path)

    # MMdet/COCO post process
    # test_file_path = "/Users/kevin/Downloads/test.json"
    # test_result_path = "/Users/kevin/Downloads/convnext_316.bbox.json"
    # ResultUtil.submit_format_converter(test_file_path, test_result_path)

    # YOlOX post process
    # sample_path = "/Users/kevin/Downloads/DCIC2022_dataset/submit_example.csv"
    # label_path = "/Users/kevin/Downloads/label"
    # ResultUtil.yolofCoco2Yolo(sample_path, label_path)

    # YOlOX post process with 2cls label
    # sample_path = "/Users/kevin/Downloads/DCIC2022_dataset/submit_example.csv"
    # label_path = "/Users/kevin/Downloads/label"
    # ResultUtil.yolofCoco2YoloWithClsLabel(sample_path, label_path)

    #wbf post process
    # cur_path = os.getcwd()
    # sample_path = os.path.join(cur_path, "submit_sample.csv")
    sample_path = "/data/code/utils/submit_sample.csv"
    # label_path = os.path.join(cur_path, "../../user_data/intermediate_results/wbf_output")
    label_path = "/data/user_data/intermediate_results/wbf_output"
    ResultUtil.wbf2submission(sample_path, label_path)
