%% ��������ǰ224����Ƶ����
global SAMPLING_FACTOR
%% ������ر�ǩ�ļ��ĺ���
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

%% ������
% �������б�ǩ�ļ�
gTruth = loadLabelFiles();

%% �������ڴ洢����Ƶ��ȡͼ����ļ���
output_dir = fullfile(pwd, 'Training_data_V');
if ~isfolder(output_dir)
    mkdir(output_dir);
end

%% ����ѵ�����ݼ�
% ��ȡ��ע���ݼ��е�����Դ����
numDataSources = numel(gTruth);
disp(['gTruth ��Ŀ����: ', num2str(numDataSources)]);

% ����ǰ224����Ƶ
disp('����ǰ224����Ƶ...');
gTruthPart1 = gTruth(1:224);
samplingFactor = ones(1, 224) * SAMPLING_FACTOR;
trainingData1 = objectDetectorTrainingData(gTruthPart1, ...
    'SamplingFactor', samplingFactor, ...
    'WriteLocation', output_dir);

% �����һ���ֽ��
tempFile = fullfile(pwd, 'temp_training_data.mat');
save(tempFile, 'trainingData1', 'output_dir');

disp('ǰ224����Ƶ������ɲ��ѱ�����ʱ�ļ�');