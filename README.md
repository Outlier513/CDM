# **深度学习与传统认知诊断模型在教育评估中的对比分析**

## Description

这是我们实现**深度学习与传统认知诊断模型在教育评估中的对比分析**所构建的项目，包含IRT， DINA，MIRT，NCD在ASSIST和Junyi两个数据集上训练和验证的结果。

### Dataset

| Statistics         | ASSIST  | Junyi   |
| ------------------ | ------- | ------- |
| Students           | 2493    | 10000   |
| Exercises          | 17746   | 835     |
| Knowledge concepts | 123     | 835     |
| Response records   | 267,415 | 353,835 |

#### ASSIST

log_data.json

+ 学生练习记录
+ 来源于：Source: https://github.com/bigdata-ustc/Neural_Cognitive_Diagnosis-NeuralCD/blob/master/data/log_data.json

train_set.json

+ 训练数据文件

test_set.josn

+ 验证测试文件

#### Junyi

+ log_data.json

  + 学生练习记录
  + 来源于：Source: https://github.com/bigdata-ustc/RCD/blob/main/data/junyi/log_data.json

  train_set.json

  + 训练数据文件

  test_set.josn

  + 验证测试文件

## File structure



## Environment Settings

- torch version 2.1.1+cu121

## Usage

以使用ASSIST数据集和NCD模型作为例子

```
python train.py -d cuda:0 -m NCD -ds ASSIST
```

其他的参数可以在`config.yaml`中进行设置