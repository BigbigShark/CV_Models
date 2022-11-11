# -*- coding: utf-8 -*-
'''将.txt格式标注信息转换为.xml格式，存入到datasets/Annotations目录下'''
from lxml.etree import Element, SubElement, tostring

all_classes = ['bird', 'car', 'dog', 'lizard', 'turtle']

def txt_to_xml(img_file, txt_path, xml_file):
    clas = []
    with open(txt_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            list = line.split(" ")
            clas.append(list)
    for i in range(len(clas)):
        if i > 179:
            break
        xml_name = "{}{}.xml".format(class_name, i+1)  # bird000188.xml
        xml_path = xml_file + '\\' + xml_name
        file_name = "{}{}.JPEG".format(class_name, i+1)

        node_root = Element('annotation')
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = class_name
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = file_name
        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = '128'
        node_height = SubElement(node_size, 'height')
        node_height.text = '128'
        node_depth = SubElement(node_size, 'depth')
        node_depth.text = '3'


        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = class_name
        node_pose=SubElement(node_object, 'pose')
        node_pose.text="Unspecified"
        node_truncated=SubElement(node_object, 'truncated')
        node_truncated.text="truncated"
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(clas[i][1])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(clas[i][2])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(clas[i][3])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(clas[i][4])
        xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
        img_newxml = xml_path
        file_object = open(img_newxml, 'wb')
        file_object.write(xml)
        file_object.close()

def main(class_name):
    # 图像文件夹所在位置
    img_file = r".\tiny_vid\{}".format(class_name)
    # 标注文件夹所在位置
    txt_path = r".\tiny_vid\{}_gt.txt".format(class_name)
    # txt转化成xml格式后存放的文件夹
    xml_file = r".\Annotations"

    txt_to_xml(img_file, txt_path, xml_file)

if __name__ == "__main__":
    for class_name in all_classes:
        main(class_name)
