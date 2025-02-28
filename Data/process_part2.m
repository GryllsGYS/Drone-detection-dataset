%% ��������ʣ����Ƶ����
global SAMPLING_FACTOR
% ���¼��ر�Ҫ�ı���
tempFile = fullfile(pwd, 'temp_training_data.mat');
if ~exist(tempFile, 'file')
    error('�Ҳ�����ʱ�ļ�����������process_part1.m');
end
load(tempFile);

% ���¼��ر�ǩ�ļ�
function gTruth = loadLabelFiles()
% ��ʼ���յ�groundTruth����
gTruth = [];

% �ɻ����ݣ�001-059��
for i = 1:59
    try
        % �����ļ���
        filename = sprintf('V_AIRPLANE_%03d_LABELS.mat', i);
        % �����ļ�
        gt = load(filename);
        % ��groundTruth��ӵ�����
        gTruth = [gTruth; gt.gTruth];
    catch
        warning('�޷������ļ�: %s', filename);
    end
end

% �������ݣ�001-051��
for i = 1:51
    try
        filename = sprintf('V_BIRD_%03d_LABELS.mat', i);
        gt = load(filename);
        gTruth = [gTruth; gt.gTruth];
    catch
        warning('�޷������ļ�: %s', filename);
    end
end

% ���˻����ݣ�001-114��
for i = 1:114
    try
        filename = sprintf('V_DRONE_%03d_LABELS.mat', i);
        gt = load(filename);
        gTruth = [gTruth; gt.gTruth];
    catch
        warning('�޷������ļ�: %s', filename);
    end
end

% ֱ�������ݣ�001-061��
for i = 1:61
    try
        filename = sprintf('V_HELICOPTER_%03d_LABELS.mat', i);
        gt = load(filename);
        gTruth = [gTruth; gt.gTruth];
    catch
        warning('�޷������ļ�: %s', filename);
    end
end

% ѡ������ı�ǩ
gTruth = selectLabels(gTruth, {'AIRPLANE','BIRD','DRONE','HELICOPTER'});
end

gTruth = loadLabelFiles();

% ����ʣ����Ƶ
disp('����ʣ����Ƶ...');
gTruthPart2 = gTruth(225:end);
samplingFactor = ones(1, numel(gTruthPart2)) * SAMPLING_FACTOR;
trainingData2 = objectDetectorTrainingData(gTruthPart2, ...
    'SamplingFactor', samplingFactor, ...
    'WriteLocation', output_dir);

% ����ڶ����ֽ��
tempFile2 = fullfile(pwd, 'temp_training_data2.mat');
save(tempFile2, 'trainingData2');

disp('ʣ����Ƶ������ɲ��ѱ�����ʱ�ļ�');