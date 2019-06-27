function KNNdatgingTest
%%
clc
clear
close all
%%
data = load('datingTestSet2.txt');
dataMat = data(:,1:3);
labels = data(:,4);
len = size(dataMat,1); %����
k = 4;
error = 0;
% �������ݱ���
Ratio = 0.1;
numTest = Ratio * len;
% ��һ������
maxV = max(dataMat);
minV = min(dataMat);
range = maxV-minV;
newdataMat = (dataMat-repmat(minV,[len,1]))./(repmat(range,[len,1]));

% ����
for i = 1:numTest
    classifyresult = KNN(newdataMat(i,:),newdataMat(numTest:len,:),labels(numTest:len,:),k);
    fprintf('���Խ��Ϊ��%d  ��ʵ���Ϊ��%d\n',[classifyresult labels(i)])
    if(classifyresult~=labels(i))
        error = error+1;
    end
end
  fprintf('׼ȷ��Ϊ��%f\n',1-error/(numTest))
end
