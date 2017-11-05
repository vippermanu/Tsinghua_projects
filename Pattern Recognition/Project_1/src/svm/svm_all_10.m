clear all, close all, clc

% read data
[d_3,l_3] = xlsread('dataset3.xlsx');
[d_4,l_4] = xlsread('dataset4.xlsx');

% train the model
mdl = svmtrain(d_3,l_3,'kernel_function','linear');

% classification
pred = svmclassify(mdl,d_4);

% calculate error rate
num = 0;
for i = 1:length(l_4)
   if pred{i} ~= l_4{i}
       num = num + 1;
   end
end
err = num/length(l_4);

disp('error rate:'),disp(err);