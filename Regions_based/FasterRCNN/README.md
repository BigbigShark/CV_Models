## FasterRCNN目录结构

│  classes_index.json//目标类别，注意把0留给背景<br>
│  predict.py//预测模型
│  train.py//训练模型
│  README.md
│  
├─backbone
│  │  fasterrcnn_resnet50_fpn_coco.pth
│  │  feature_pyramid.py
│  │  resnet50.pth
│  │  resnet50_fpn.py
│  │  __init__.py
│  │  
│  └─__pycache__
│          
├─datasets//训练数据集
│  │  put2jpegfile.py
│  │  split_data.py
│  │  txt2xml.py
│  │  
│  ├─Annotations
│  └─JPEGImages
├─info_of_train//训练过程的信息
├─network
│  │  boxes.py
│  │  det_utils.py
│  │  faster_rcnn_framework.py
│  │  image_list.py
│  │  roi_head.py
│  │  rpn_function.py
│  │  transform.py
│  │  __init__.py
│  │  
│  └─__pycache__
│          
├─save_weights//模型权重，将下载好的模型权重放在该目录下，并起名model.pth
│      model.pth
│      
├─test_imgs//预测模型使用的测试图片，以.jpg或.JPEG结尾
├─test_result//预测模型输出的结果
├─train_result//训练的结果，包括loss和mAP
│      loss_and_lr20221111-221624.png
│      mAP.png
│      
└─train_utils
    │  coco_eval.py
    │  coco_utils.py
    │  distributed_utils.py
    │  draw_box_utils.py
    │  group_by_aspect_ratio.py
    │  myDataset.py
    │  plot_curve.py
    │  train_val_utils.py
    │  transforms.py
    │  __init__.py
    │  
    └─__pycache__

## 模型权重

链接：https://pan.baidu.com/s/1sBP9ej6PwKQlJf0oWnKo1Q 
提取码：t2i4