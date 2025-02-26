%% 简介
% 本程序用于从视频和标签文件创建用于训练的数据集




%% 加载所有标签文件
% 初始化空的groundTruth数组
gTruth = [];

% 飞机数据（001-059）
for i = 1:59
    try
        % 生成文件名
        filename = sprintf('V_AIRPLANE_%03d_LABELS.mat', i);
        % 加载文件
        gt = load(filename);
        % 将groundTruth添加到数组
        gTruth = [gTruth; gt.gTruth];
    catch
        warning('无法加载文件: %s', filename);
    end
end

% 鸟类数据（001-051）
for i = 1:51
    try
        filename = sprintf('V_BIRD_%03d_LABELS.mat', i);
        gt = load(filename);
        gTruth = [gTruth; gt.gTruth];
    catch
        warning('无法加载文件: %s', filename);
    end
end

% 无人机数据（001-114）
for i = 1:114
    try
        filename = sprintf('V_DRONE_%03d_LABELS.mat', i);
        gt = load(filename);
        gTruth = [gTruth; gt.gTruth];
    catch
        warning('无法加载文件: %s', filename);
    end
end

% 直升机数据（001-061）
for i = 1:61
    try
        filename = sprintf('V_HELICOPTER_%03d_LABELS.mat', i);
        gt = load(filename);
        gTruth = [gTruth; gt.gTruth];
    catch
        warning('无法加载文件: %s', filename);
    end
end

% 选择所需的标签
gTruth = selectLabels(gTruth, {'AIRPLANE','BIRD','DRONE','HELICOPTER'});


%% 创建用于存储从视频提取图像的文件夹
if isfolder(fullfile('Training_data_V'))
    cd Training_data_V
else
    mkdir Training_data_V
end 
addpath('Training_data_V');


%% 生成训练数据集
% 采样因子为1表示从视频中提取所有帧
trainingData = objectDetectorTrainingData(gTruth,...
    'SamplingFactor', 1, ...  
    'WriteLocation','Training_data_V');


%% 保存训练数据集
save('Training_data_V.mat', "trainingData")


%% 结果说明
% 现在应该得到以下内容:
% 1. 包含从视频提取的图像的文件夹(Training_data_V)
% 2. 一个.mat文件，行数与图像数量相同，包含5列:
%    - 第1列：图像路径
%    - 第2-5列：四个类别(飞机、鸟类、无人机、直升机)的边界框信息



