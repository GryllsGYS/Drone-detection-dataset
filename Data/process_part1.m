%% 处理并采样前224个视频样本
global SAMPLING_FACTOR
%% 定义加载标签文件的函数
function gTruth = loadLabelFiles()
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
end

%% 主程序
% 加载所有标签文件
gTruth = loadLabelFiles();

%% 创建用于存储从视频提取图像的文件夹
output_dir = fullfile(pwd, 'Training_data_V');
if ~isfolder(output_dir)
    mkdir(output_dir);
end

%% 生成训练数据集
% 获取标注数据集中的数据源数量
numDataSources = numel(gTruth);
disp(['gTruth 条目数量: ', num2str(numDataSources)]);

% 处理前224个视频
disp('处理前224个视频...');
gTruthPart1 = gTruth(1:224);
samplingFactor = ones(1, 224) * SAMPLING_FACTOR;
trainingData1 = objectDetectorTrainingData(gTruthPart1, ...
    'SamplingFactor', samplingFactor, ...
    'WriteLocation', output_dir);

% 保存第一部分结果
tempFile = fullfile(pwd, 'temp_training_data.mat');
save(tempFile, 'trainingData1', 'output_dir');

disp('前224个视频处理完成并已保存临时文件');