## FasterRCNN目录结构

│  classes_index.json//目标类别，注意把0留给背景<br>
│  predict.py//预测模型<br>
│  train.py//训练模型<br>
│  README.md<br>
│  <br>
├─backbone<br>
│  │  fasterrcnn_resnet50_fpn_coco.pth<br>
│  │  feature_pyramid.py<br>
│  │  resnet50.pth<br>
│  │  resnet50_fpn.py<br>
│  │  __init__.py<br>
│  │  <br>
│  └─__pycache__<br>
│          <br>
├─datasets//训练数据集<br>
│  │  put2jpegfile.py<br>
│  │  split_data.py<br>
│  │  txt2xml.py<br>
│  │  
│  ├─Annotations<br>
│  └─JPEGImages<br>
├─info_of_train//训练过程的信息<br>
├─network<br>
│  │  boxes.py<br>
│  │  det_utils.py<br>
│  │  faster_rcnn_framework.py<br>
│  │  image_list.py<br>
│  │  roi_head.py<br>
│  │  rpn_function.py<br>
│  │  transform.py<br>
│  │  __init__.py<br>
│  │  <br>
│  └─__pycache__<br>
│          <br>
├─save_weights//模型权重，将下载好的模型权重放在该目录下，并起名model.pth<br>
│      model.pth<br>
│      <br>
├─test_imgs//预测模型使用的测试图片，以.jpg或.JPEG结尾<br>
├─test_result//预测模型输出的结果<br>
├─train_result//训练的结果，包括loss和mAP<br>
│      loss_and_lr20221111-221624.png<br>
│      mAP.png<br>
│      <br>
└─train_utils<br>
    │  coco_eval.py<br>
    │  coco_utils.py<br>
    │  distributed_utils.py<br>
    │  draw_box_utils.py<br>
    │  group_by_aspect_ratio.py<br>
    │  myDataset.py<br>
    │  plot_curve.py<br>
    │  train_val_utils.py<br>
    │  transforms.py<br>
    │  __init__.py<br>
    │  <br>
    └─__pycache__<br>

## 模型权重

链接：https://pan.baidu.com/s/1sBP9ej6PwKQlJf0oWnKo1Q <br>
提取码：t2i4
