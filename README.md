# Sparkify


### 项目背景和总览
》Sparkify是Udacity数据科学家纳米学位毕业项目提供的的一个虚构的音乐流应用程序;它提供付费和免费的音乐服务，客户可以在两种服务之间切换，并且他们可以随时取消订阅。

》给定的客户数据集是有12GB，因此标准的分析和机器学习工具在这里无法使用，因为它将不适合计算机的内存（数据分析和建模需要占用超过32GB甚至更多计算机的内存，会使得单机系统奔溃）。

》执行这种分析的安全方法是使用大数据工具，如Apache Spark，这是最快的大数据工具之一。

》为本项目提供了小型、中型和大型的三种尺寸的数据集。我选择使用了小型尺寸的数据集，以便在本地单机上处理。

###问题描述
本项目的目标是要创建一个机器学习模型在这之前预测用户取消订阅的意图（称为客户流失）。
解决问题思路：
1.数据处理：这部分我们将数据进行预处理为后续分析和建模打好基础，包含：1）数据集清洗和缺失值处理 2）EDA 3)特征工程处理

2.建立模型：这部分致力于利用机器学习在选定特征变量的基础上，预测用户取消订阅的意图；利用处理后数据集进行不同机器学习模型训练和评价

3.生成应用：将之前两部分的内容进行打包和整理，最终生成一个应用程序；并介绍使用的原理和方法

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
Through the file [`GeneralizeSparkify.py`]， 我们可以按照如下步骤进行操作:

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
