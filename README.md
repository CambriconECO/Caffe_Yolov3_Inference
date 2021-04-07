## yolov3_caffe_demo 使用

#### 介绍

此demo演示了caffe框架下的yolov3模型在MLU270上的移植流程。

本示例基于 Neuware 1.6.1 + Python2.7 版本测试通过。

相关教程链接：https://developer.cambricon.com/index/curriculum/expdetails/id/4/classid/8.html

#### 0、运行依赖下载
###### 在运行该demo前，需要下载相关依赖，Caffe_Yolov3_Inference路径下：

```shell
bash setup.sh
```

#### 1、模型结构转换

###### convert路径下：

```python
python darknet2caffe-yoloV23.py 3  yolov3.cfg  yolov3.weights  yolov3.prototxt  yolov3.caffemodel
```
###### 在转换为prototxt文件后，还需要添加后处理节点，该步骤请详细参考教程：https://developer.cambricon.com/index/curriculum/expdetails/id/4/classid/8.html

#### 2、模型量化

###### quantize路径下：

```shell
./generate_quantized_pt --ini_file convert_quantized.ini
```

#### 3、在线推理

###### online路径下：

```python
python detect.py yolov3_int8.prototxt yolov3.caffemodel
```

#### 4、离线推理

###### 生成离线模型，offline路径下：

```shell
 ./caffe genoff -model ../quantize/yolov3_int8.prototxt -weights  ../quantize/yolov3.caffemodel -mcore MLU270
```

###### 离线推理，offline/yolov3_offline_simple_demo路径下：

```
编译：bash make.sh
运行：bash run.sh
```

