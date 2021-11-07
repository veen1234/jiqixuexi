
#===================================================================================================

import numpy as np
import imageio
import cv2
from datetime import datetime
import glob
import os
import scipy.io as sio
from sklearn.decomposition import PCA
import tensorflow as tf
from Models import model_unet, model_DenseNet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#gpu_options = tf.GPUOptions(allow_growth=True)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#===================================================================================================



# 超参数
img_size = 9
classes = 10
LR = 1e-4
img_h = 550
img_w = 400
bands = 270
input_sizes = (img_size, img_size, 32)
batch_size = 64

# epochs = 10

#===================================================================================================

def makedirs(dir_path):

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print('{}: Folder creation successful: {}'.format(datetime.now().strftime('%c'), dir_path))
    else:
        print('{}: Folder already exists: {}'.format(datetime.now().strftime('%c'), dir_path))

    return dir_path

#===================================================================================================

def model_predict(model, img_data, lab_data):

    #初始化一个 0 矩阵，用于将预测结果的值赋值到 0 矩阵的对应位置
    pre_img = np.zeros((img_h, img_w), dtype='uint8')
    img = []

    # 对 25*25大小的图像进行预测
    y = np.zeros((img_h+size-1, img_w+size-1, 32))
    y[(size-1)//2:img_h+(size-1)//2, (size-1)//2:img_w+(size-1)//2, :] = img_data
    
    for i in range (img_h):
        print(i)
        for j in range (img_w):
            sub = y[i:size+i,j:size+j,:]
            
            img.append(sub)

            if (j+1) % 10 == 0:
                img = np.array(img)
                # 预测，对结果进行处理
                y_pre = model.predict(img)
                # print(y_pre.shape)
                # y_pre = np.squeeze(y_pre, axis = 0)
                y_pre = np.argmax(y_pre, axis = -1)
                y_pre = y_pre.astype('uint8')
    
                # 将预测结果的值赋值到 0 矩阵的对应位置
                pre_img[i, j-9:j+1] = y_pre
                img = []
                        

    # 计算准确率
    acc = np.mean(np.equal(lab_data, pre_img))
    return acc, pre_img

#=========================================================================================

if __name__ == '__main__':
    """
    主函数
    """
    """
    加载图像信息
    """
    # 预测图像和标签
    pre_name = 'indianPines'
    lab_name = 'indianPines_gt'
    size = 9
    test_img = []
    test_lab = []
    # 加载预测图像
    def loadData(data):
        if data == 'IndiaP':
            images = sio.loadmat('./datasets/IndiaP_RTV.mat')['fimg']
            lables = sio.loadmat('./datasets/indian_pines_corrected_gt.mat')['indian_pines_gt']
        elif data == 'paviaU':
            images = sio.loadmat('./datasets/paviaU.mat')['paviaU']
            lables = sio.loadmat('./datasets/paviaU_gt.mat')['paviaU_gt']
        elif data == 'salinas':
            images = sio.loadmat('./datasets/salinas_corrected.mat')['salinas_corrected']
            lables = sio.loadmat('./datasets/salinas_gt.mat')['salinas_gt']
        else:
            print('No Model!')
        return images, lables


    # 加载数据
    img_data, img_lab = loadData('IndiaP')
    # img_data, img_lab = loadData('paviaU')
    # img_data, img_lab = loadData('salinas')
    print(img_data.shape)
                                                                                                                                                                                           
    # 最大最小归一化
    def split(image_data, num):
        image_data_list = [] #创建列表 保存规范化后的数据
        for i in range(num): #创建循环 注意是从0开始
            B = cv2.split(image_data)[i]
            B_normalization = ((B - np.min(B)) / (np.max(B) - np.min(B)) * 1).astype('float32')
            image_data_list.append(B_normalization)
        data = cv2.merge(image_data_list)
        return data
    img_data = split(img_data, bands)
    
    #PCA降维
    img_data = img_data.reshape((img_h*img_w ,bands))
    sklearn_pca = PCA(n_components = 32)
    img_data = sklearn_pca.fit_transform(img_data)
    img_data = img_data.reshape((img_h, img_w, 32))
    print(img_data.shape)
    
    # 加载标签图像
    lab_path = os.path.join(os.getcwd(), lab_name)
    lab_data = imageio.imread(lab_path)

    """
    加载模型信息
    """
    model_name_list = ['UNet', 'Densenet']
    model_name = model_name_list[4]
    
    # 导入模型
    if model_name == model_name_list[0]:
         model = model_unet.unet(input_size=input_sizes, num_class=classes, model_summary=True)
         print('==========================Start {}!======================================='.format(model_name))

         model = model_DenseNet.densenet(input_size=input_sizes, num_class=classes, model_summary=True)
         print('==========================Start {}!======================================='.format(model_name))
    else:
        print('No Model!')


    #===================================================================================================

    # 预测结果的存放目录
    y_pre_savepath = makedirs(os.path.join(os.getcwd(), 'model', ))

    # 获取模型列表
    model_list = glob.glob(os.path.join(os.getcwd(), 'model', '*.hdf5'))

    print('{}: Find {} model\n'.format(datetime.now().strftime('%c'), len(model_list)))

    for i in range(len(model_list)):
        
        # 获取模型名字
        model_name = (model_list[i].split('\\')[-1]).split('.hdf5')[0]
        print('\n{}: Statr No.{:<3d} model, the name is {}'.format(datetime.now().strftime('%c'), i+1, model_name))

        # 加载模型参数
        model.load_weights(model_list[i])

        # 预测结果
        # accuracy, predict_result = model_predict(model, img_data, lab_data, img_size)
        accuracy, predict_result = model_predict(model, img_data, lab_data)
        print('\n{}: The accuracy rate is {:<0.5f}, the model name is {}'.format(datetime.now().strftime('%c'), accuracy, model_name))

        # 保存预测结果
        imageio.imwrite(os.path.join(y_pre_savepath, model_name+'_acc-'+str(round(accuracy, 4))+'.tif'), predict_result)
        print('{}: Predict the success of image {}\n'.format(datetime.now().strftime('%c'), pre_name))
#=========================================================================================




























