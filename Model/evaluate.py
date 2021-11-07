# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 09:22:20 2019

@author: zhangyonghui
"""
import tensorflow as tf
import os
import numpy as np
import imageio
import json
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score

# pred_img = imageio.imread(os.path.join(os.getcwd(),  'newtest_svm.tif'))
# pred_img = pred_img.transpose([1,2,0])
# print(pred_img.shape)

"""
本文件主要实现以下功能：
1、评估模型，包括准确率、精准率、召回率、F1分数
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# 类名
class_name = ['Corn11111',
              'Cotton111',
              'Sesame111',
              'Broad1111',
              'Narrow111',
              'Rice11111',
              'Water1111',
              'Roads1111',
              'Mixed1111']

#========================================================================================================================================

def calculation(y_label, y_pre, row, col):
    '''
    本函数主要计算以下评估标准的值：
    1、精准率
    2、召回率
    3、F1分数
    '''

    # 转成列向量
    y_label = np.reshape(y_label, (row*col, 1))
    y_pre = np.reshape(y_pre, (row*col, 1))
    # y_pre = np.reshape(y_pre, (3145728, 1))
    
    y_label.astype('float64')
    y_pre.astype('float64')

    # 精准率
    precision = precision_score(y_label, y_pre, average=None)

    # 召回率
    recall = recall_score(y_label, y_pre, average=None)

    # F1
    f1 = f1_score(y_label, y_pre, average=None)
    
    # kappa
    kappa = cohen_kappa_score(y_label, y_pre)
    
    #IoU
    iou = (precision*recall)/(precision + recall - (precision*recall) )
    
    return precision, recall, f1, kappa, iou

#========================================================================================================================================

def estimate(y_label, y_pred, model_hdf5_name, class_name, dirname):
    '''
    本函数主要实现以下功能：
    1、计算准确率
    2、将各种评估指标存成一个json格式的txt文件
    @parameter:
        y_label:标签 
        y_pred:预测结果
        model_hdf5_name:模型名
        class_name:类型
        dirname:存放路径
    '''

    # 准确率
    acc = np.mean(np.equal(y_label, y_pred) + 0)
    print('=================================================================================================')
    print('The estimate result of {}.hdf5 are as follows:'.format(model_hdf5_name))
    print('The acc of {} is {}'.format(model_hdf5_name, acc))

    precision, recall, f1, kappa, iou = calculation(y_label, y_pred, y_label.shape[0], y_label.shape[1])

    for i in range(len(class_name)):

        print('{}    F1: {:.5f}, Precision: {:.5f}, Recall: {:.5f}, kappa: {:.5f}, iou: {:.5f}'.format(class_name[i], f1[i], precision[i], recall[i], kappa, iou[i]))
    print('=================================================================================================')

    if len(f1) == len(class_name):

        result = {}

        for i in range(len(class_name)):

            result[class_name[i]] = []
            tmp = {}
            tmp['Recall'] = str(round(recall[i], 5))
            tmp['Precision'] = str(round(precision[i], 5))
            tmp['F1'] = str(round(f1[i], 5))
            result[class_name[i]].append(tmp)

        result['Model Name'] = [model_hdf5_name]
        result['Accuracy'] = str(round(acc, 5))
        result['kappa'] = str(kappa)
        result['iou'] = str(iou)

        # 写入txt
        with open(os.path.join(dirname, model_hdf5_name+'.txt'), 'a', encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False))

    else:
        print("======================================>Estimate error!===========================================")

if __name__ == '__main__':

    # model_hdf5_name = r'DenseNet-83-0.9866_acc-0.9856'
    model_hdf5_name = r'DenseNetSE-03-0.9634_acc-0.9637'
    # 读取真值图和预测图
    true_img = imageio.imread(os.path.join(os.getcwd(), 'WHU-Hi-LongKou_gt.tif'))
    # pred_img = imageio.imread(os.path.join(os.getcwd(), 'svm',  model_hdf5_name+'.tif'))
    pred_img = imageio.imread(os.path.join(os.getcwd(), 'Saved_models', 'test', 'DenseNetSE', '原图test', model_hdf5_name+'.tif'))
    
    # pred_img = pred_img +1
    dirname = os.path.join(os.path.join(os.getcwd()))
    estimate(true_img, pred_img, model_hdf5_name, class_name, dirname)
























