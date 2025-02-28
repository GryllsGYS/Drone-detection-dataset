%% 处理并采样剩余视频样本
global SAMPLING_FACTOR
% 重新加载必要的变量
tempFile = fullfile(pwd, 'temp_training_data.mat');
if ~exist(tempFile, 'file')
    error('找不到临时文件。请先运行process_part1.m');
end
load(tempFile);

% 重新加载标签文件
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

gTruth = loadLabelFiles();

% 处理剩余视频
disp('处理剩余视频...');
gTruthPart2 = gTruth(225:end);
samplingFactor = ones(1, numel(gTruthPart2)) * SAMPLING_FACTOR;
trainingData2 = objectDetectorTrainingData(gTruthPart2, ...
    'SamplingFactor', samplingFactor, ...
    'WriteLocation', output_dir);

% 保存第二部分结果
tempFile2 = fullfile(pwd, 'temp_training_data2.mat');
save(tempFile2, 'trainingData2');

disp('剩余视频处理完成并已保存临时文件');