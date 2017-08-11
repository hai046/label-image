# tensorflow 例子
这都是官方tensorflow例子

不过自己稍微改动一下
## 根据图片识别物体
官方例子 `https://www.tensorflow.org/tutorials/image_recognition`
修改后的本地例子 `tutorials/image/imagenet/classify_image.py` 是执行的具体py

`imagenet_synset_to_human_label_map_CN.txt`  这个是百度翻译后的，有的不准，需要持续自己找出来重新翻译了一下

相关阅读：
http://www.cnblogs.com/neopenx/p/4480701.html

### 后续

https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/label_image/README.md 比本项目实现识别率更高

### 更新2017年08月11日10:11
例子 `image_retraining`

目前添加了五个分类
```
chounan
meinv
shuaige
xinggan
chounv
dachangtui

```


一共收集到6类 图片用百度上搜索

感觉可以用来评估 某一个人的颜值

效果

|图片|结果|
|-|-|
|![](http://dynamic-image.yesky.com/740x-/uploadImages/2015/183/03/1VWS4F8TN76I.jpg)|长腿美女[0.62927],性感美女[0.34004],帅哥[0.01739],美女[0.01269],丑男[0.00036]|
|![](http://img.tupianzj.com/uploads/allimg/151228/9-15122R02H0.jpg)| 性感美女[0.96357],长腿美女[0.03571],美女[0.00063],帅哥[0.00007],丑男[0.00001]|
|![](https://b-ssl.duitang.com/uploads/item/201403/29/20140329133232_EsQtf.thumb.700_0.jpeg)| 性感美女[0.96357],长腿美女[0.03571],美女[0.00063],帅哥[0.00007],丑男[0.00001]|


### 更新2017年08月08日11:50

image_retraining 目录下可以自己训练模型  自己识别  66666

自己用 retrain 训练自己的模型


数据目前准备不是很多 只准备了2类

先训练，然后获取
```
meinv (score = 0.91679)
zhengexin (score = 0.08321)

```


```
./meinv/2l62MqnzYNMD8s41TXBPxw.jpg
./meinv/64DrAUrEKxQD8s41TXBPxw.jpg
./meinv/8RUe-EXobVMD8s41TXBPxw.jpg
./meinv/946D_jy2NWMD8s41TXBPxw.jpg
./meinv/_B-O-aT2OI4D8s41TXBPxw.jpg
./meinv/B1Jrv0WtWYMD8s41TXBPxw.jpg
./meinv/bfGJ_gFTILID8s41TXBPxw.jpg
./meinv/cz_Um2bm3_kD8s41TXBPxw.jpg
./meinv/dIU4ICmM2pID8s41TXBPxw.jpg
./meinv/G2o023etqx0D8s41TXBPxw.jpg
./meinv/gbWE8dbqB5AD8s41TXBPxw.jpg
./meinv/hkEcICqDCqkD8s41TXBPxw.jpg
./meinv/i-qEklELwuYD8s41TXBPxw.jpg
./meinv/i3Fy-Zz3GXQD8s41TXBPxw.jpg
./meinv/iJd0vMzHtoAD8s41TXBPxw.jpg
./meinv/lwX9E0q2DhMD8s41TXBPxw.jpg
./meinv/NcuTlfpfEmcD8s41TXBPxw.jpg
./meinv/p2m6MqcGDUwD8s41TXBPxw.jpg
./meinv/PokCnYjJIX0D8s41TXBPxw.jpg
./meinv/QRTpuKka_CsD8s41TXBPxw.jpg
./meinv/uQAMrmNw52MD8s41TXBPxw.jpg
./meinv/v3nUXT_kdloD8s41TXBPxw.jpg
./meinv/w4KYTuSFP9kD8s41TXBPxw.jpg
./meinv/waEzeyTdn-kD8s41TXBPxw.jpg
./meinv/wQw_GqH_PPgD8s41TXBPxw.jpg
./meinv/yLAqn9O2_zkD8s41TXBPxw.jpg
./meinv/ZO9v2ArwJ9gD8s41TXBPxw.jpg
./zhengexin/-iazHhuX4B4D8s41TXBPxw.jpg
./zhengexin/4EspAGhgRAID8s41TXBPxw.jpg
./zhengexin/6aDtGp9aNlkD8s41TXBPxw.jpg
./zhengexin/7Gi3fdzvc1YD8s41TXBPxw.jpg
./zhengexin/e2EYqTjeG4oD8s41TXBPxw.jpg
./zhengexin/F92grA0hxy4D8s41TXBPxw.jpg
./zhengexin/fI9jPXmdtNMD8s41TXBPxw.jpg
./zhengexin/gzDv9_HcbKUD8s41TXBPxw.jpg
./zhengexin/HtGXn-vsV7ID8s41TXBPxw.jpg
./zhengexin/JLvjl5w-tx4D8s41TXBPxw.jpg
./zhengexin/JOrMaUGINkMD8s41TXBPxw.jpg
./zhengexin/KWOHktCKODQD8s41TXBPxw.jpg
./zhengexin/LjrnypUZrgoD8s41TXBPxw.jpg
./zhengexin/p1DSEanzXJkD8s41TXBPxw.jpg
./zhengexin/PEzO1VyRQe4D8s41TXBPxw.jpg
./zhengexin/QAax0T_ouNsD8s41TXBPxw.jpg
./zhengexin/qBwgCnxhLhMD8s41TXBPxw.jpg
./zhengexin/QeUQQKB7VLMD8s41TXBPxw.jpg
./zhengexin/SKNZjg3CrjcD8s41TXBPxw.jpg
./zhengexin/SLS-c7k5XzYD8s41TXBPxw.jpg
./zhengexin/Ud8Ay6g86ooD8s41TXBPxw.jpg
./zhengexin/Vidr3ZvZht8D8s41TXBPxw.jpg
./zhengexin/WRoXmuFq7UkD8s41TXBPxw.jpg
./zhengexin/XPigdt18mL0D8s41TXBPxw.jpg
./zhengexin/yzY4IeW-t2kD8s41TXBPxw.jpg
./zhengexin/Z9HwNTacohID8s41TXBPxw.jpg

```