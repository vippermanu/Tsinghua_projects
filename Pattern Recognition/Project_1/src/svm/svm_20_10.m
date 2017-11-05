clear all, close all, clc

% read data
[d_3,l_3] = xlsread('dataset3.xlsx');
[d_4,l_4] = xlsread('dataset4.xlsx');

iter = 100;
error = zeros(1,iter);

for it = 1:iter
    % pick 20 samples
    f_num = unidrnd(469,10,1);
    m_num = unidrnd(485,10,1);
    m_num = m_num + 469;
    
    data = [d_3(f_num,:);d_3(m_num,:)];
    label = [l_3(f_num);l_3(m_num)];

    % train the model
    mdl = svmtrain(data,label,'kernel_function','linear');
    % rbf < linear < mlp, others can't converge

    % classification
    pred = svmclassify(mdl,d_4);

    % calculate error rate
    num = 0;
    for i = 1:length(l_4)
       if pred{i} ~= l_4{i}
           num = num + 1;
       end
    end
    error(it) = num/length(l_4);
end

min_err = min(error);
mean_err = mean(error);
disp('minimum error rate:'),disp(min_err);
disp('mean error rate:'),disp(mean_err);