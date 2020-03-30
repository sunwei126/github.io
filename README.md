# Sparkify


### 项目背景和总览
》一个名为“Sparkify”的虚拟公司，它提供付费和免费的音乐服务，客户可以在两种服务之间切换，并且他们可以随时取消订阅。
》给定的客户数据集是有12GB，因此标准的分析和机器学习工具在这里无法使用，因为它将不适合计算机的内存（数据分析和建模需要占用超过32GB甚至更多计算机的内存，会使得单机系统奔溃）。
》执行这种分析的安全方法是使用大数据工具，如Apache Spark，这是最快的大数据工具之一。
》在本项目中，我们只使用原始数据集的128MB片段

**需解决的问题**是要创建一个机器学习模型在这之前预测用户取消订阅的意图（称为客户流失）。

### 项目中文件夹和文件

* **文件夹/Folders**
  * [Trained_models](等待上传到github): 该文件夹包含已训练好的模型
    * DecisionTreeClassifier.model
    * GradientBoostedTrees.model
    * LogisticRegression.model
    * MultilayerPerceptronClassifier.model
    * RandomForestClassifier.model
    * model_LogReg.model
	
* **文件/Files**
  * [Sparkify.ipynb](等待上传到github): Udacity workspace 的代码过程
  * [GeneralizeSparkify.py](等待上传到github): 根据workspace代码生成的应用程序
  * [Sparkify.html](等待上传到github): Udacity workspace 的代码过生成的HTML文件
  * [README.md](等待上传到github): 本文件
  * [saved_user_dataset_pd.CSV](等待上传到github): 用于机器学习的训练数据集


### 未包含的原始文件
* mini_sparkify_event_data.json: 原始jason文件(超出 GitHub 容量限制)

### 使用软件和资源
- Python 3.6.x 
- pySpark 2.4.x 
- matplotlib 3.03 
- pandas 0.23 
- jupyter 

### 生成分析工具
Through the file [`GeneralizeSparkify.py`](https://github.com/drnesr/Sparkify/blob/master/GeneralizeSparkify.py), we can do the following:

**第一步，导入工具和数据**

```python
from GeneralizeSparkify import load_clean_transfer
```

**假设我们有名为“new_data.json”的新数据，写下以下命令：**

```python
load_clean_transfer('new_data.json', save_as='new_dat_extraction')
```

此命令将：
1.从数据源读取数据，
2.清洗数据
3.将已清洗的数据集另存为新名称“new-dat-extraction.CSV”`

**第二步，导入模型和分析**

```python
from GeneralizeSparkify import load_ml_dataset, get_train_test_features, apply_model
```

**1. 加载提取的数据new_data_extraction.CSV`**

```python
ml_ds = load_ml_dataset(saved_as='new_data_extraction.CSV')
```

**2 获取训练、测试数据集和特征变量名称**

```python
train, test, features_labels = get_train_test_features(ml_ds)
```

***3.对数据应用机器学习模型***

***可以创建模型：***

```python
apply_model(train, test, features_labels, model_name='GBT',save_as='NewGBT.model')
```

***也可以使用已训练的模型***

```python
apply_model(train, test, features_labels,model_name='LR', load_from_existing='LogisticRegression.model')
```

### 更多了解
请邮件至sunwei126_2009@sina.com交流本项目中更加详细内容

