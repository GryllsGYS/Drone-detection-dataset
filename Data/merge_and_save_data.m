%% �ϲ����������ݲ��������ս��

% ������ʱ�ļ�
tempFile1 = fullfile(pwd, 'temp_training_data.mat');
tempFile2 = fullfile(pwd, 'temp_training_data2.mat');

if ~exist(tempFile1, 'file') || ~exist(tempFile2, 'file')
    error('�Ҳ�����ʱ�ļ�����������process_part1.m��process_part2.m');
end

load(tempFile1, 'trainingData1', 'output_dir');
load(tempFile2, 'trainingData2');

% �ϲ����
trainingData = [trainingData1; trainingData2];

% ������ת��Ϊ���鲢����
if exist('trainingData', 'var')
    trainingDataArray = table2array(trainingData);
    save(fullfile(pwd, 'Training_data_V_array.mat'), 'trainingDataArray');
    disp('�ɹ�����������ѵ�����ݣ�');
else
    error('�޷�����ѵ������');
end

% ɾ����ʱ�ļ�
delete(tempFile1);
delete(tempFile2);
disp('��ʱ�ļ���ɾ��');

%% ���˵��
disp('������ɣ����ɵ����ݰ���:');
disp(['1. ��������Ƶ��ȡ��ͼ����ļ���: ', output_dir]);
disp('2. Training_data_V_array.mat�ļ����������б�ע���ݣ�����:');
disp('   - ��1�У�ͼ��·��');
disp('   - ��2-5�У��ĸ����(�ɻ������ࡢ���˻���ֱ����)�ı߽����Ϣ');