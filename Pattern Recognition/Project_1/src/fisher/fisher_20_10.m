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
    
    f = d_3(f_num,:);
    m = d_3(m_num,:);

    % train the model
    prior = [0.5,0.5];

    u_m = mean(m);
    u_f = mean(f);

    s_m = cov(m)*(10-1);  % matlab normalizes covariance with N-1
    s_f = cov(f)*(10-1);

    sw = s_m + s_f;
    sb = (u_m-u_f)' * (u_m-u_f);

    w = sw \ (u_m-u_f)';

    u_m1 = w' * u_m';
    u_f1 = w' * u_f';

    w0 = 0.5 * (u_m1+u_f1);
    policy = u_m1 > u_f1;

    % predict
    pred = zeros(length(l_4),1);
    for i = 1:length(l_4)
        data = d_4(i,:);
        pred(i) = fisher_judge(data,w,w0,policy);
    end

    % calculate error rate
    num = 0;
    for i = 1:length(l_4)
       if (pred(i) == 1 && l_4{i} == 'F') || (pred(i) == 0 && l_4{i} == 'M') 
           num = num + 1;
       end
    end
    error(it) = num/length(l_4);
end

min_err = min(error);
mean_err = mean(error);
disp('minimum error rate:'),disp(min_err);
disp('mean error rate:'),disp(mean_err);