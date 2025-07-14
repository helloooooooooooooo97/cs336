# 莺尾花数据集简介

莺尾花（Iris）数据集是机器学习领域中最经典、最常用的多分类数据集之一，由英国统计学家Fisher于1936年整理发布。该数据集包含150条样本数据，分为三类：山鸢尾（Iris-setosa）、变色鸢尾（Iris-versicolor）和维吉尼亚鸢尾（Iris-virginica），每类各50个样本。

## 数据集特征

每个样本有4个特征，均为连续型变量：

- **萼片长度（sepal length，cm）**
- **萼片宽度（sepal width，cm）**
- **花瓣长度（petal length，cm）**
- **花瓣宽度（petal width，cm）**

目标变量为鸢尾花的类别（setosa、versicolor、virginica）。

## 应用场景

- 多分类问题的入门练习
- 特征可视化与降维（如PCA）
- 各类机器学习算法的对比与调优

## 典型任务

1. 数据可视化与探索性分析
2. 特征工程与数据预处理
3. 构建分类模型（如决策树、KNN、SVM等）
4. 评估模型性能（准确率、混淆矩阵等）

## 参考

Iris数据集可通过`sklearn.datasets.load_iris()`直接获取，适合初学者进行数据分析和建模实践。