%% 从视频和标签创建数据集的主脚本
% 该脚本按顺序执行数据处理的各个步骤
% 采样为每SAMPLING_FACTOR帧采样一次
global SAMPLING_FACTOR
SAMPLING_FACTOR = 1;
%% 步骤1：处理前224个样本
disp('开始执行第1步：处理前224个视频样本...');
process_part1;
disp('第1步完成！');

%% 步骤2：处理剩余样本
disp('开始执行第2步：处理剩余视频样本...');
process_part2;
disp('第2步完成！');

%% 步骤3：合并数据集并保存
disp('开始执行第3步：合并数据并保存...');
merge_and_save_data;
disp('第3步完成！');

disp('整个处理流程已完成！');