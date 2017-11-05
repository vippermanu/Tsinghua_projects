clear all, close all

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
    
    f = d_3(f_num,5:6);
    m = d_3(m_num,5:6);

    % train the model
    prior = [0.5,0.5];

    u_m = mean(m);
    u_f = mean(f);

    cov_m = cov(m);
    cov_f = cov(f);

    % predict
    pred = zeros(length(l_4),1);
    for i = 1:length(l_4)
        data = d_4(i,5:6);
        g_m = bayes_judge(data,u_m,cov_m,2,prior(1));
        g_f = bayes_judge(data,u_f,cov_f,2,prior(2));
        if g_m > g_f
            pred(i) = 1;
        else
            pred(i) = 0;
        end
    end

    % calculate error rate
    show = it == iter;
    num = 0;
    if show
        for i = 1:length(l_4)
            if (pred(i) == 1 && l_4{i} == 'F') || (pred(i) == 0 && l_4{i} == 'M') 
                num = num + 1;
                if l_4{i} == 'F'
                    plot(d_4(i,5),d_4(i,6),'ks','MarkerFaceColor','b',...
                        'MarkerSize',6);
                    hold on;
                else
                    plot(d_4(i,5),d_4(i,6),'kx','LineWidth',2,...
                        'MarkerSize',6);
                    hold on;
                end
            else
                if l_4{i} == 'F'
                    plot(d_4(i,5),d_4(i,6),'ko','MarkerFaceColor','y',...
                        'MarkerSize',6);
                    hold on;
                else
                    plot(d_4(i,5),d_4(i,6),'k+','LineWidth',2,...
                        'MarkerSize',6);
                    hold on;
                end
            end
        end
    else
        for i = 1:length(l_4)
           if (pred(i) == 1 && l_4{i} == 'F') || (pred(i) == 0 && l_4{i} == 'M') 
               num = num + 1;
           end
        end
    end
    error(it) = num/length(l_4);
end

min_err = min(error);
mean_err = mean(error);
disp('minimum error rate:'),disp(min_err);
disp('mean error rate:'),disp(mean_err);