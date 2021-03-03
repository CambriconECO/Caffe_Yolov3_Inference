## yolov3_caffe_demo 使用

#### 1、模型结构转换

###### yolov3_caffe_demo/convert路径下：

```python
python darknet2caffe-yoloV23.py 3  yolov3.cfg  yolov3.weights  yolov3.prototxt  yolov3.caffemodel
```

#### 2、模型量化

###### yolov3_caff_demo/quantize路径下：

```shell
./generate_quantized_pt --ini_file convert_quantized.ini
```

#### 3、在线推理

###### yolov3_caff_demo/online路径下：

```python
python detect.py yolov3_int8.prototxt yolov3.caffemodel
```

#### 4、离线推理

###### 生成离线模型，在yolov3_caffe_demo/offline路径下：

```shell
 ./caffe genoff -model ../quantize/yolov3_int8.prototxt -weights       ../quantize/yolov3.caffemodel -mcore MLU270
```

###### 离线推理，在yolov3_caffe_demo/offline/yolov3_offline_simple_demo路径下：

```
编译：bash make.sh
运行：bash run.sh
```

