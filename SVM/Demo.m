
%%%% P. Duan, et al,%  "Noise-Robust Hyperspectral Image Classification via Multi-Scale Total Variation"
% IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.
clc
clear
close all
addpath ('functions')
addpath (genpath('libsvm-3.22'))
addpath (genpath('KPCA1'))
%% load original image
path='.\Datasets\';
inputs = 'IndiaP';%145*145*200/10249/16 
location = [path,inputs];
load (location);

%%% size of image 
[no_lines, no_rows, no_bands] = size(img);
GroundT=GroundT';
load (['.\training_indexes\in_1.mat'])
%% Spectral dimension Reduction
 img2=average_fusion(img,20);
 OA=[];AA=[];kappa=[];CA=[];
for i=1:10
    indexes=dph(:,i);
%% Normalization
no_bands=size(img2,3);
fimg=reshape(img2,[no_lines*no_rows no_bands]);
[fimg] = scale_new(fimg);
fimg=reshape(fimg,[no_lines no_rows no_bands]);
%% KPCA
 fimg =kpca(fimg, 1000,30, 'Gaussian',20);%'Gaussian'

%% save datasets
%savePath = 'D:\jiang\trainModel\test\data\IndiaP_RTV.mat';
%save('IndiaP_RTV.mat','fimg');
%save('IndiaP_G', 'GroundT');
%% SVM classification
    fimg = ToVector(fimg);
    fimg = fimg';
    fimg=double(fimg);
%%%
train_SL = GroundT(:,indexes);
train_samples = fimg(:,train_SL(1,:))';
train_labels= train_SL(2,:)';
%%%
test_SL = GroundT;
test_SL(:,indexes) = [];
test_samples = fimg(:,test_SL(1,:))';
test_labels = test_SL(2,:)';
% Normalizing Training and original img 
[train_samples,M,m] = scale_func(train_samples);
[fimg ] = scale_func(fimg',M,m);

% Selecting the paramter for SVM
[Ccv Gcv cv cv_t]=cross_validation_svm(train_labels,train_samples);
% Training using a Gaussian RBF kernel
parameter=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv);
model=svmtrain(train_labels,train_samples,parameter);
% Testing
Result = svmpredict(ones(no_lines*no_rows,1),fimg,model); %%%
% Evaluation
GroudTest = double(test_labels(:,1));
ResultTest = Result(test_SL(1,:),:);
[OA_i,AA_i,kappa_i,CA_i]=confusion(GroudTest,ResultTest);%
OA=[OA OA_i];
AA=[AA AA_i];
kappa=[kappa kappa_i];
CA=[CA CA_i];
Result = reshape(Result,no_lines,no_rows);
VClassMap=label2colord(Result,'india');
figure,imshow(VClassMap);
end
OA_std=std(OA);OA_mean=mean(OA);
AA_std=std(AA);AA_mean=mean(AA);
K_std=std(kappa);kappa_mean=mean(kappa);
disp('%%%%%%%%%%%%%%%%%%% Classification Results of MSTV Method %%%%%%%%%%%%%%%%')
disp(['OA',' = ',num2str(OA_mean),' ||  ','AA',' = ',num2str(AA_mean),'  ||  ','Kappa',' = ',num2str(kappa_mean)])
