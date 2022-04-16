import os
import json
import argparse
import time

args = []
parser1 = argparse.ArgumentParser()
parser1.add_argument('--json_path', default='/data/user_data/intermediate_results/conv_output/swin_large_2cls_full_exp1.bbox.json',type=str, help="input: coco format(json)")
parser1.add_argument('--save_path', default='/data/user_data/intermediate_results/output_models', type=str, help="specify where to save the output dir of labels")
arg1 = parser1.parse_args()
args.append(arg1)

parser2 = argparse.ArgumentParser()
parser2.add_argument('--json_path', default='/data/user_data/intermediate_results/conv_output/swin_large_continue_exp2.bbox.json',type=str, help="input: coco format(json)")
parser2.add_argument('--save_path', default='/data/user_data/intermediate_results/output_models', type=str, help="specify where to save the output dir of labels")
arg2 = parser2.parse_args()
args.append(arg2)

parser3 = argparse.ArgumentParser()
parser3.add_argument('--json_path', default='/data/user_data/intermediate_results/conv_output/cascade_convnext_x_exp1.bbox.json',type=str, help="input: coco format(json)")
parser3.add_argument('--save_path', default='/data/user_data/intermediate_results/output_models', type=str, help="specify where to save the output dir of labels")
arg3 = parser3.parse_args()
args.append(arg3)

parser4 = argparse.ArgumentParser()
parser4.add_argument('--json_path', default='/data/user_data/intermediate_results/conv_output/cascade_convnext_x_2cls_full_640_exp6.bbox.json',type=str, help="input: coco format(json)")
parser4.add_argument('--save_path', default='/data/user_data/intermediate_results/output_models', type=str, help="specify where to save the output dir of labels")
arg4 = parser4.parse_args()
args.append(arg4)

parser5 = argparse.ArgumentParser()
parser5.add_argument('--json_path', default='/data/user_data/intermediate_results/conv_output/cascade_convnext_x_1cls_640_exp2.bbox.json',type=str, help="input: coco format(json)")
parser5.add_argument('--save_path', default='/data/user_data/intermediate_results/output_models', type=str, help="specify where to save the output dir of labels")
arg5 = parser5.parse_args()
args.append(arg5)

parser6 = argparse.ArgumentParser()
parser6.add_argument('--json_path', default='/data/user_data/intermediate_results/conv_output/cascade_convnext_x_2cls_full_aug_256_exp9.bbox.json',type=str, help="input: coco format(json)")
parser6.add_argument('--save_path', default='/data/user_data/intermediate_results/output_models', type=str, help="specify where to save the output dir of labels")
arg6 = parser6.parse_args()
args.append(arg6)

parser7 = argparse.ArgumentParser()
parser7.add_argument('--json_path', default='/data/user_data/intermediate_results/conv_output/cascade_convnext_x_1cls_full_exp4.bbox.json',type=str, help="input: coco format(json)")
parser7.add_argument('--save_path', default='/data/user_data/intermediate_results/output_models', type=str, help="specify where to save the output dir of labels")
arg7 = parser7.parse_args()
args.append(arg7)

parser8 = argparse.ArgumentParser()
parser8.add_argument('--json_path', default='/data/user_data/intermediate_results/conv_output/cascade_convnext_x_2cls_full_exp5.bbox.json',type=str, help="input: coco format(json)")
parser8.add_argument('--save_path', default='/data/user_data/intermediate_results/output_models', type=str, help="specify where to save the output dir of labels")
arg8 = parser8.parse_args()
args.append(arg8)


def json_filter(input_dict):
    # with open(path, "r") as json_str:
    #     input_dict = json.loads(json_str.read())

    # Filter python objects with list comprehensions
    output_dict = [x for x in input_dict if x["category_id"] == 1]

    # Transform python object back into json
    # output_json = json.dumps(output_dict)

    # with open('json_data.json', 'w') as outfile:
    #     outfile.write(output_json)
    return output_dict


def convert(box):
    x1 = box[0]
    y1 = box[1]
    x2 = x1 + box[2]
    y2 = y1 + box[3]
    return (x1, y1, x2, y2)

if __name__ == '__main__':
    for arg in args:
        # cur_path = os.getcwd()
        
        json_file = arg.json_path

        # json_file = os.path.join(cur_path, json_file)
        # media_path = os.path.join(cur_path, arg.save_path)

        media_path = arg.save_path
        current_time = time.localtime()

        ana_txt_save_path = os.path.join(media_path, time.strftime("%m%d%H%M%S", current_time))
        names_path = '/data/code/utils/test.json'
        # names_path = os.path.join(cur_path, "test.json")

        #print(json_file)
        data1 = json.load(open(json_file, 'r'))
        
        data = json_filter(data1)

        id2name = json.load(open(names_path, 'r'))

        #print(data)
        if not os.path.exists(ana_txt_save_path):
            os.makedirs(ana_txt_save_path)

        id_map = {}
        for img in id2name['images']:
            value = img['file_name']
            key = img['id']
            id_map[key] = value

        count = 0
        for piece in data:
            id = piece['image_id']
            bbox = piece['bbox']
            score = piece['score']
            Bbox = convert(bbox)
            txt_name = id_map[id].replace('.jpg', '') + '.txt'
            f_txt = open(os.path.join(ana_txt_save_path, txt_name), 'a')
            if score >= 0.7:
                f_txt.write("%s %s %s %s %s\n" % (Bbox[0], Bbox[1], Bbox[2], Bbox[3], score))
            f_txt.close()
