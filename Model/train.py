import numpy as np
from keras.utils import np_utils

from Models import model_unet, model_DenseNet
import scipy.io as sio
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import glob
import os

model_name = 'DenseNet'
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
#加载数据
images, lables = loadData('IndiaP')
#images, lables = loadData('paviaU')
#images, lables = loadData('salinas')
images = images.reshape((-1, 30))
print(images.shape)
#进行归一化
transfer = MinMaxScaler()
x = transfer.fit_transform(images)
images = x.reshape((145, 145, 30))  #将尺寸还原成(145,145,30)

#train_image, val_image, train_lable, val_lable = train_test_split(images, lables, test_size=0.1, random_state=0)
def padwithzeros(X,margin=2):
    newX=np.zeros((X.shape[0]+2*margin,X.shape[1]+2*margin,X.shape[2]))
    x_offset=margin
    y_offset=margin
    newX[x_offset:X.shape[0]+x_offset,y_offset:X.shape[1]+y_offset,:]=X
    return newX


def creatCube(X, y, windowsize=25, removeZeroLabels=True):
    margin = int((windowsize - 1) / 2)  # margin=12
    zeroPaddedX = padwithzeros(X, margin=margin)

    patchesData = np.zeros((X.shape[0] * X.shape[1], windowsize, windowsize, X.shape[2]))  # (145*145,25,25,30)
    patchesLabels = np.zeros(X.shape[0] * X.shape[1])
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):  # (12,145-12)=(12,132)
        for c in range(margin, zeroPaddedX.shape[1] - margin):  # (12,145-12)=(12,132)
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels

images,lables=creatCube(images,lables,windowsize=9)  #打包成Cube

lables=np_utils.to_categorical(lables)

IMAGE_SIZE = 9
LR = 0.0001
epochs = 10
batch_size = 2
classes = 16
K = 10
input_size=(IMAGE_SIZE, IMAGE_SIZE, 30)
#导入模型
model = model_DenseNet.densenet(input_size=input_size, num_class=classes, model_summary=True)
#model = MCNN.mcnn(input_size=input_size, num_class=classes)
def model_set(model_name, k):
    save_dir = os.path.join(os.getcwd(), 'model')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 模型保存路径
    save_model_path = os.path.join(save_dir, model_name + '-K-' + str(k) + '-{epoch:02d}-{val_loss:.4f}.hdf5')
    model_checkpoint = ModelCheckpoint(save_model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto',
                                       period=1, save_weights_only=True)

    callback_lists = [model_checkpoint]

    return callback_lists
# K 折交叉验证
skf = StratifiedKFold(n_splits=K)

# 设置交叉验证时必须的参数 y
y = len(lables)
y = np.zeros(y)

n = 0
for train, valid in skf.split(images, y):
    n += 1
    print('------------------------------K = {:<2d}({})------------------------------'.format(n, K))

    # 模型设置
    callback_lists = model_set(model_name, n)

    train_img = images[train, :, :]
    # train_lab = lables[train, :, :, :]
    train_lab = lables[train, :]

    valid_img = images[valid, :, :]
    # valid_lab = lables[valid, :, :, :]
    valid_lab = lables[valid, :]

    # 训练
    model.fit(train_img,
              train_lab,
              validation_data=(valid_img, valid_lab),
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              callbacks=callback_lists)