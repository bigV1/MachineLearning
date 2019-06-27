%% KNN
clear all
clc
%% data
trainData = [1.0,2.0;1.2,0.1;0.1,1.4;0.3,3.5];
trainClass = [1,1,2,2];
testData = [0.5,2.3];
k = 3;

%% distance
row = size(trainData,1);
col = size(trainData,2);
test = repmat(testData,row,1);
dis = zeros(1,row);
for i = 1:row
    diff = 0;
    for j = 1:col
        diff = diff + (test(i,j) - trainData(i,j)).^2;
    end
    dis(1,i) = diff.^0.5;
end

%% sort
jointDis = [dis;trainClass];
sortDis= sortrows(jointDis');
sortDisClass = sortDis';

%% find
class = sort(2:1:k);
member = unique(class);
num = size(member);

max = 0;
for i = 1:num
    count = find(class == member(i));
    if count > max
        max = count;
        label = member(i);
    end
end

disp('最终的分类结果为：');
fprintf('%d\n',label)