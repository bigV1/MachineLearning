function handWritingTest
%%
clc
clear
close all
%% ��ȡĿ¼�µ�����txt�ļ�����
d = dir(['digits/trainingDigits/' '*.txt']); % struct ����
dircell = struct2cell(d); %cell ����
trainSetLen = size(dircell,2);
K = 4;
dataSize = 1024;
trainLabels = zeros(trainSetLen,1);
trainSet = [];
simpleTrainSet = zeros(1,dataSize);
simpleTestSet = zeros(1,dataSize);

%% ��������
fprintf('loading data...')
for i = 1:trainSetLen
    trainName =  dircell(1,i);
    trainFilename = cell2mat(trainName);
    trainLabels(i) = str2num(trainFilename(1));

    fid = fopen(['digits/trainingDigits/' trainFilename],'r');
    traindata = fscanf(fid,'%s');
    for j = 1:dataSize
        simpleTrainSet(j) =  str2num(traindata(j));
    end
    trainSet = [trainSet ; simpleTrainSet];
    fclose(fid);
end

d = dir(['digits/testDigits/' '*.txt']); % struct ����
dircell = struct2cell(d); %cell ����
testSetLen = size(dircell,2);
error = 0;
%% ��������
for k = 1:testSetLen
    testName =  dircell(1,k);
    testFilename = cell2mat(testName);
    testLabels = str2num(testFilename(1));

    fid = fopen(['digits/testDigits/' testFilename],'r');
    testdata = fscanf(fid,'%s');
    for j = 1:dataSize
        simpleTestSet(j) =  str2num(testdata(j));
    end
    classifyResult = KNN(simpleTestSet,trainSet,trainLabels,K);
    fprintf('ʶ������Ϊ��%d  ��ʵ����Ϊ��%d\n' , [classifyResult , testLabels])
    if(classifyResult~=testLabels)
        error = error+1;
    end
    fclose(fid);
end

fprintf('ʶ��׼ȷ��Ϊ��%f\n',1-error/testSetLen)

end
