# -*- coding: utf-8 -*-
"""

Created on Mon Oct  9 11:22:46 2023

@author: user
"""
import xml.etree.ElementTree as ET
import os
import argparse


# 0 -> inside_person
# 1 -> outside
# 2 -> ignore_IN
# 3 -> ignore_OUT
# 4 -> ignore_IO
# 5 -> time
# 6 -> inside_car
# 7 -> cash_region
# 8 -> queue_region
# 9 -> cahsier_region

def single_xml_to_txt(xml_file, jpg_file, class_names, args):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    #保存txt文件路径
    txt_file = (os.path.basename(xml_file)[:-4] + '.txt')
    with open(txt_file, 'w') as tf:
        tf.write('d' + ' ' + args.d + '\n')
        tf.write('t' + ' ' + args.t + '\n')
        tf.write('p' + ' ' + args.p + '\n')
        tf.write('c' + ' ' + args.c + '\n')
        tf.write('o' + ' ' + args.o + '\n')
        tf.write('f' + ' ' + args.f + '\n')
        tf.write('r' + ' ' + args.r + '\n')
        for member in root.findall('object'):
            #从xml获取图像的宽和高
            picture_width = int(root.find('size')[0].text)
            picture_height = int(root.find('size')[1].text)
            class_name = member[0].text
            #类名对应的index
            class_num = class_names.index(class_name)
            box_x_min = int(member[4][0].text)  # 左上角横坐标
            box_y_min = int(member[4][1].text)  # 左上角纵坐标
            box_x_max = int(member[4][2].text)  # 右下角横坐标
            box_y_max = int(member[4][3].text)  # 右下角纵坐标
            
            xy = f"{box_x_min},{box_y_min} {box_x_max},{box_y_max}"
            # 转成相对位置和宽高（所有值处于0~1之间）
#            x_center = (box_x_min + box_x_max) / (2 * picture_width)
#            y_center = (box_y_min + box_y_max) / (2 * picture_height)
#            width = (box_x_max - box_x_min) / picture_width
#            height = (box_y_max - box_y_min) / picture_height
            #print(class_num, x_center, y_center, width, height)
            tf.write(str(class_num) + ' ' + xy + '\n')
            print(str(class_num) + ' ' + xy + '\n')
        

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default="00:00:00",
                               help='hh:mm:ss')
    parser.add_argument('-d', type=str, default="",
                               help='dd/mm/yyyy')
    parser.add_argument('-p', type=str, default="0.8",
                               help='human conf')
    parser.add_argument('-c', type=str, default="0.9",
                               help='car conf')
    parser.add_argument('-o', type=str, default="0.9",
                               help='outside conf')
    parser.add_argument('-f', type=str, default="1",
                               help='skip frame')
    parser.add_argument('-r', type=str, default="1",
                               help='show ratio depend on 720/1080') # 1 -> 720 | 2-> 1080  
    parser.add_argument('-m', type=str, default="1",
                               help='model mode depend on Object Counting/Transaction Counting') # 1 -> object counting | transaction counting 
    
    opt = parser.parse_args()
    return opt
    

if __name__ == "__main__":
    opt = parse_opt()
    print(f"Date: {opt.d}\nStart time: {opt.t}\nPerson conf: {opt.p}\nCar conf: {opt.c}\nOutside conf: {opt.o}")
    if opt.r == "1":
        resolution = "720x1280"
    elif opt.r == "2":
        resolution = "1080x1980"
    else:
        opt.r = 1
        resolution = "Wrong number option not exists. Default 720x1280"
        
    if opt.m == "1":
        mode = "Object Counting"
    elif opt.m == "2":
        mode = "Transaction Counting"
    else:
        opt.m = 1
        mode = "Wrong number mode not exits. Default Object Counting Mode"

    print(f"Frame rate: {opt.f}\nResolution: {resolution}\n\n")
    class_names = ["inside_person", "outside", "ignore_IN", "ignore_OUT","ignore_IO", "time", "inside_car",
                   "pay_region", "queue_region", "cashier_region"]
    single_xml_to_txt("out.xml", "out.jpg", class_names, opt)




