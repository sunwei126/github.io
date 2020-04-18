# Sparkify


### ��Ŀ����������
��Sparkify��Udacity���ݿ�ѧ������ѧλ��ҵ��Ŀ�ṩ�ĵ�һ���鹹��������Ӧ�ó���;���ṩ���Ѻ���ѵ����ַ��񣬿ͻ����������ַ���֮���л����������ǿ�����ʱȡ�����ġ�
�������Ŀͻ����ݼ�����12GB����˱�׼�ķ����ͻ���ѧϰ�����������޷�ʹ�ã���Ϊ�������ʺϼ�������ڴ棨���ݷ����ͽ�ģ��Ҫռ�ó���32GB���������������ڴ棬��ʹ�õ���ϵͳ��������
��ִ�����ַ����İ�ȫ������ʹ�ô����ݹ��ߣ���Apache Spark���������Ĵ����ݹ���֮һ��
��Ϊ����Ŀ�ṩ��С�͡����ͺʹ��͵����ֳߴ�����ݼ�����ѡ��ʹ����С�ͳߴ�����ݼ����Ա��ڱ��ص����ϴ���

###��������
����Ŀ��Ŀ����Ҫ����һ������ѧϰģ������֮ǰԤ���û�ȡ�����ĵ���ͼ����Ϊ�ͻ���ʧ����
�������˼·��
1.���ݴ����ⲿ�����ǽ����ݽ���Ԥ����Ϊ���������ͽ�ģ��û�����������1�����ݼ���ϴ��ȱʧֵ���� 2��EDA 3)�������̴���
2.����ģ�ͣ��ⲿ�����������û���ѧϰ��ѡ�����������Ļ����ϣ�Ԥ���û�ȡ�����ĵ���ͼ�����ô�������ݼ����в�ͬ����ѧϰģ��ѵ��������
3.����Ӧ�ã���֮ǰ�����ֵ����ݽ��д����������������һ��Ӧ�ó��򣻲�����ʹ�õ�ԭ��ͷ���

### ��Ŀ���ļ��к��ļ�

* **�ļ���/Folders**
  * [Trained_models](�ȴ��ϴ���github): ���ļ��а�����ѵ���õ�ģ��
    * DecisionTreeClassifier.model
    * GradientBoostedTrees.model
    * LogisticRegression.model
    * MultilayerPerceptronClassifier.model
    * RandomForestClassifier.model
    * model_LogReg.model
	
* **�ļ�/Files**
  * [Sparkify.ipynb](�ȴ��ϴ���github): Udacity workspace �Ĵ������
  * [GeneralizeSparkify.py](�ȴ��ϴ���github): ����workspace�������ɵ�Ӧ�ó���
  * [Sparkify.html](�ȴ��ϴ���github): Udacity workspace �Ĵ�������ɵ�HTML�ļ�
  * [README.md](�ȴ��ϴ���github): ���ļ�
  * [saved_user_dataset_pd.CSV](�ȴ��ϴ���github): ���ڻ���ѧϰ��ѵ�����ݼ�


### δ������ԭʼ�ļ�
* mini_sparkify_event_data.json: ԭʼjason�ļ�(���� GitHub ��������)

### ʹ���������Դ
- Python 3.6.x 
- pySpark 2.4.x 
- matplotlib 3.03 
- pandas 0.23 
- jupyter 

### ���ɷ�������
Through the file [`GeneralizeSparkify.py`]�� ���ǿ��԰������²�����в���:

**��һ�������빤�ߺ�����**

```python
from GeneralizeSparkify import load_clean_transfer
```

**������������Ϊ��new_data.json���������ݣ�д���������**

```python
load_clean_transfer('new_data.json', save_as='new_dat_extraction')
```

�������
1.������Դ��ȡ���ݣ�
2.��ϴ����
3.������ϴ�����ݼ����Ϊ�����ơ�new-dat-extraction.CSV��`

**�ڶ���������ģ�ͺͷ���**

```python
from GeneralizeSparkify import load_ml_dataset, get_train_test_features, apply_model
```

**1. ������ȡ������new_data_extraction.CSV`**

```python
ml_ds = load_ml_dataset(saved_as='new_data_extraction.CSV')
```

**2 ��ȡѵ�����������ݼ���������������**

```python
train, test, features_labels = get_train_test_features(ml_ds)
```

***3.������Ӧ�û���ѧϰģ��***

***���Դ���ģ�ͣ�***

```python
apply_model(train, test, features_labels, model_name='GBT',save_as='NewGBT.model')
```

***Ҳ����ʹ����ѵ����ģ��***

```python
apply_model(train, test, features_labels,model_name='LR', load_from_existing='LogisticRegression.model')
```

### �����˽�
���ʼ���sunwei126_2009@sina.com��������Ŀ�и�����ϸ����
