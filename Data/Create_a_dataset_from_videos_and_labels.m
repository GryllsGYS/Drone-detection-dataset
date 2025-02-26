%% ���
% ���������ڴ���Ƶ�ͱ�ǩ�ļ���������ѵ�������ݼ�




%% �������б�ǩ�ļ�
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


%% �������ڴ洢����Ƶ��ȡͼ����ļ���
if isfolder(fullfile('Training_data_V'))
    cd Training_data_V
else
    mkdir Training_data_V
end 
addpath('Training_data_V');


%% ����ѵ�����ݼ�
% ��������Ϊ1��ʾ����Ƶ����ȡ����֡
trainingData = objectDetectorTrainingData(gTruth,...
    'SamplingFactor', 1, ...  
    'WriteLocation','Training_data_V');


%% ����ѵ�����ݼ�
save('Training_data_V.mat', "trainingData")


%% ���˵��
% ����Ӧ�õõ���������:
% 1. ��������Ƶ��ȡ��ͼ����ļ���(Training_data_V)
% 2. һ��.mat�ļ���������ͼ��������ͬ������5��:
%    - ��1�У�ͼ��·��
%    - ��2-5�У��ĸ����(�ɻ������ࡢ���˻���ֱ����)�ı߽����Ϣ



