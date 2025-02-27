%% 简介
% 本程序用于从视频和标签文件创建用于训练的数据集




%% 加载所有标签文件
% 初始化空的groundTruth数组
gTruth = [];

% 飞机数据（001-059）
for i = 1:2
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
% 清理未使用的变量并释放内存
clear gt filename i;
disp('飞机数据加载完成，已释放内存');
% 强制垃圾回收
java.lang.System.gc();

% 鸟类数据（001-051）
for i = 1:2
    try
        filename = sprintf('V_BIRD_%03d_LABELS.mat', i);
        gt = load(filename);
        gTruth = [gTruth; gt.gTruth];
    catch
        warning('无法加载文件: %s', filename);
    end
end
% 清理未使用的变量并释放内存
clear gt filename i;
disp('鸟类数据加载完成，已释放内存');
% 强制垃圾回收
java.lang.System.gc();

% 无人机数据（001-114）
for i = 1:2
    try
        filename = sprintf('V_DRONE_%03d_LABELS.mat', i);
        gt = load(filename);
        gTruth = [gTruth; gt.gTruth];
    catch
        warning('无法加载文件: %s', filename);
    end
end
% 清理未使用的变量并释放内存
clear gt filename i;
disp('无人机数据加载完成，已释放内存');
% 强制垃圾回收
java.lang.System.gc();

% 直升机数据（001-061）
for i = 1:2
    try
        filename = sprintf('V_HELICOPTER_%03d_LABELS.mat', i);
        gt = load(filename);
        gTruth = [gTruth; gt.gTruth];
    catch
        warning('无法加载文件: %s', filename);
    end
end
% 清理未使用的变量并释放内存
clear gt filename i;
disp('直升机数据加载完成，已释放内存');
% 强制垃圾回收
java.lang.System.gc();

% 选择所需的标签
gTruth = selectLabels(gTruth, {'AIRPLANE','BIRD','DRONE','HELICOPTER'});


%% 创建用于存储从视频提取图像的文件夹
output_dir = fullfile(pwd, 'Training_data_V');
if ~isfolder(output_dir)
    mkdir(output_dir);
end


%% 生成训练数据集
% 获取标注数据集中的数据源数量
numDataSources = 0;
try
    % 尝试获取 gTruth 中的数据源数量
    numDataSources = numel(gTruth.DataSource.Source);
    disp(['数据源数量: ', num2str(numDataSources)]);
catch
    % 如果上述方法不起作用，尝试获取 gTruth 条目的数量
    numDataSources = numel(gTruth);
    disp(['gTruth 条目数量: ', num2str(numDataSources)]);
end

try
    % 每个视频采样一个恒定比例的帧（这里设置为1，采样所有帧）
    samplingFactor = ones(1, 8) * 1;

    % 创建训练数据
    trainingData = objectDetectorTrainingData(gTruth, ...
        'SamplingFactor', samplingFactor, ...
        'WriteLocation', output_dir);
catch e
    % 详细错误信息
    disp('创建训练数据时出错:');
    disp(['错误信息: ', e.message]);

    % 尝试替代方法：逐个处理视频
    disp('尝试替代方法...');

    % 创建临时的单视频 gTruth 集合
    allTrainingData = table();

    % 如果 gTruth 是结构体数组
    if isstruct(gTruth) || iscell(gTruth)
        for i = 1:numel(gTruth)
            try
                disp(['处理视频 ', num2str(i), '/', num2str(numel(gTruth))]);
                singleGTruth = gTruth(i);

                % 为单个视频创建训练数据
                singleTrainingData = objectDetectorTrainingData(singleGTruth, ...
                    'SamplingFactor', 1, ...
                    'WriteLocation', output_dir);

                % 合并结果
                if isempty(allTrainingData)
                    allTrainingData = singleTrainingData;
                else
                    allTrainingData = [allTrainingData; singleTrainingData];
                end
            catch innerE
                warning(['处理视频 ', num2str(i), ' 时出错: ', innerE.message]);
                continue;
            end
        end

        % 如果成功处理了任何视频，将结果赋给 trainingData
        if ~isempty(allTrainingData)
            trainingData = allTrainingData;
        else
            error('无法处理任何视频');
        end
    else
        % 如果上述方法都失败，尝试直接处理没有采样因子的 gTruth
        disp('尝试不使用采样因子参数...');
        trainingData = objectDetectorTrainingData(gTruth, ...
            'WriteLocation', output_dir);
    end
end

%% 保存训练数据集
if exist('trainingData', 'var')
    trainingDataArray = table2array(trainingData);
    save(fullfile(pwd, 'Training_data_V_array.mat'), 'trainingDataArray');
    disp('成功创建并保存训练数据！');
else
    error('无法创建训练数据');
end

%% 结果说明
% 现在应该得到以下内容:
% 1. 包含从视频提取的图像的文件夹(Training_data_V)
% 2. 一个.mat文件，行数与图像数量相同，包含5列:
%    - 第1列：图像路径
%    - 第2-5列：四个类别(飞机、鸟类、无人机、直升机)的边界框信息



