clear all, close all

% read data
[d_3,l_3] = xlsread('dataset3.xlsx');
[d_4,l_4] = xlsread('dataset4.xlsx');

% train the model
prior = [0.5,0.5];
[m_index,f_index] = findindex(l_3);
m = d_3(m_index,:);
f = d_3(f_index,:);

u_m = mean(m);
u_f = mean(f);

cov_m = cov(m);
cov_f = cov(f);

% predict
pred = zeros(length(l_4),1);
for i = 1:length(l_4)
    data = d_4(i,:);
    g_m = bayes_judge(data,u_m,cov_m,10,prior(1));
    g_f = bayes_judge(data,u_f,cov_f,10,prior(2));
    if g_m > g_f
        pred(i) = 1;
    else
        pred(i) = 0;
    end
end

% calculate error rate
num = 0;
for i = 1:length(l_4)
   if (pred(i) == 1 && l_4{i} == 'F') || (pred(i) == 0 && l_4{i} == 'M') 
       num = num + 1;
   end
end
err = num/length(l_4);

disp('error rate:'),disp(err);