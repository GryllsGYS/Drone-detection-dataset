%% 合并两部分数据并保存最终结果

% 加载临时文件
tempFile1 = fullfile(pwd, 'temp_training_data.mat');
tempFile2 = fullfile(pwd, 'temp_training_data2.mat');

if ~exist(tempFile1, 'file') || ~exist(tempFile2, 'file')
    error('找不到临时文件。请先运行process_part1.m和process_part2.m');
end

load(tempFile1, 'trainingData1', 'output_dir');
load(tempFile2, 'trainingData2');

% 合并结果
trainingData = [trainingData1; trainingData2];

% 将数据转换为数组并保存
if exist('trainingData', 'var')
    trainingDataArray = table2array(trainingData);
    save(fullfile(pwd, 'Training_data_V_array.mat'), 'trainingDataArray');
    disp('成功创建并保存训练数据！');
else
    error('无法创建训练数据');
end

% 删除临时文件
delete(tempFile1);
delete(tempFile2);
disp('临时文件已删除');

%% 结果说明
disp('处理完成！生成的内容包括:');
disp(['1. 包含从视频提取的图像的文件夹: ', output_dir]);
disp('2. Training_data_V_array.mat文件，包含所有标注数据，其中:');
disp('   - 第1列：图像路径');
disp('   - 第2-5列：四个类别(飞机、鸟类、无人机、直升机)的边界框信息');