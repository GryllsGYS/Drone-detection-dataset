%% ���
% ���������ڴ���Ƶ�ͱ�ǩ�ļ���������ѵ�������ݼ�




%% �������б�ǩ�ļ�
% ��ʼ���յ�groundTruth����
gTruth = [];

% �ɻ����ݣ�001-059��
for i = 1:2
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
% ����δʹ�õı������ͷ��ڴ�
clear gt filename i;
disp('�ɻ����ݼ�����ɣ����ͷ��ڴ�');
% ǿ����������
java.lang.System.gc();

% �������ݣ�001-051��
for i = 1:2
    try
        filename = sprintf('V_BIRD_%03d_LABELS.mat', i);
        gt = load(filename);
        gTruth = [gTruth; gt.gTruth];
    catch
        warning('�޷������ļ�: %s', filename);
    end
end
% ����δʹ�õı������ͷ��ڴ�
clear gt filename i;
disp('�������ݼ�����ɣ����ͷ��ڴ�');
% ǿ����������
java.lang.System.gc();

% ���˻����ݣ�001-114��
for i = 1:2
    try
        filename = sprintf('V_DRONE_%03d_LABELS.mat', i);
        gt = load(filename);
        gTruth = [gTruth; gt.gTruth];
    catch
        warning('�޷������ļ�: %s', filename);
    end
end
% ����δʹ�õı������ͷ��ڴ�
clear gt filename i;
disp('���˻����ݼ�����ɣ����ͷ��ڴ�');
% ǿ����������
java.lang.System.gc();

% ֱ�������ݣ�001-061��
for i = 1:2
    try
        filename = sprintf('V_HELICOPTER_%03d_LABELS.mat', i);
        gt = load(filename);
        gTruth = [gTruth; gt.gTruth];
    catch
        warning('�޷������ļ�: %s', filename);
    end
end
% ����δʹ�õı������ͷ��ڴ�
clear gt filename i;
disp('ֱ�������ݼ�����ɣ����ͷ��ڴ�');
% ǿ����������
java.lang.System.gc();

% ѡ������ı�ǩ
gTruth = selectLabels(gTruth, {'AIRPLANE','BIRD','DRONE','HELICOPTER'});


%% �������ڴ洢����Ƶ��ȡͼ����ļ���
output_dir = fullfile(pwd, 'Training_data_V');
if ~isfolder(output_dir)
    mkdir(output_dir);
end


%% ����ѵ�����ݼ�
% ��ȡ��ע���ݼ��е�����Դ����
numDataSources = 0;
try
    % ���Ի�ȡ gTruth �е�����Դ����
    numDataSources = numel(gTruth.DataSource.Source);
    disp(['����Դ����: ', num2str(numDataSources)]);
catch
    % ������������������ã����Ի�ȡ gTruth ��Ŀ������
    numDataSources = numel(gTruth);
    disp(['gTruth ��Ŀ����: ', num2str(numDataSources)]);
end

try
    % ÿ����Ƶ����һ���㶨������֡����������Ϊ1����������֡��
    samplingFactor = ones(1, 8) * 1;

    % ����ѵ������
    trainingData = objectDetectorTrainingData(gTruth, ...
        'SamplingFactor', samplingFactor, ...
        'WriteLocation', output_dir);
catch e
    % ��ϸ������Ϣ
    disp('����ѵ������ʱ����:');
    disp(['������Ϣ: ', e.message]);

    % ����������������������Ƶ
    disp('�����������...');

    % ������ʱ�ĵ���Ƶ gTruth ����
    allTrainingData = table();

    % ��� gTruth �ǽṹ������
    if isstruct(gTruth) || iscell(gTruth)
        for i = 1:numel(gTruth)
            try
                disp(['������Ƶ ', num2str(i), '/', num2str(numel(gTruth))]);
                singleGTruth = gTruth(i);

                % Ϊ������Ƶ����ѵ������
                singleTrainingData = objectDetectorTrainingData(singleGTruth, ...
                    'SamplingFactor', 1, ...
                    'WriteLocation', output_dir);

                % �ϲ����
                if isempty(allTrainingData)
                    allTrainingData = singleTrainingData;
                else
                    allTrainingData = [allTrainingData; singleTrainingData];
                end
            catch innerE
                warning(['������Ƶ ', num2str(i), ' ʱ����: ', innerE.message]);
                continue;
            end
        end

        % ����ɹ��������κ���Ƶ����������� trainingData
        if ~isempty(allTrainingData)
            trainingData = allTrainingData;
        else
            error('�޷������κ���Ƶ');
        end
    else
        % �������������ʧ�ܣ�����ֱ�Ӵ���û�в������ӵ� gTruth
        disp('���Բ�ʹ�ò������Ӳ���...');
        trainingData = objectDetectorTrainingData(gTruth, ...
            'WriteLocation', output_dir);
    end
end

%% ����ѵ�����ݼ�
if exist('trainingData', 'var')
    trainingDataArray = table2array(trainingData);
    save(fullfile(pwd, 'Training_data_V_array.mat'), 'trainingDataArray');
    disp('�ɹ�����������ѵ�����ݣ�');
else
    error('�޷�����ѵ������');
end

%% ���˵��
% ����Ӧ�õõ���������:
% 1. ��������Ƶ��ȡ��ͼ����ļ���(Training_data_V)
% 2. һ��.mat�ļ���������ͼ��������ͬ������5��:
%    - ��1�У�ͼ��·��
%    - ��2-5�У��ĸ����(�ɻ������ࡢ���˻���ֱ����)�ı߽����Ϣ



