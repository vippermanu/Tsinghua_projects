clear all, close all, clc

% read data
[d_3,l_3] = xlsread('dataset3.xlsx');
[d_4,l_4] = xlsread('dataset4.xlsx');

% pick 2 features
dnew_3 = zeros(954,2);
dnew_4 = zeros(328,2);
dnew_3(:,:) = d_3(:,5:6);
dnew_4(:,:) = d_4(:,5:6);

% train the model
prior = [0.5,0.5];
[m_index,f_index] = findindex(l_3);
m = dnew_3(m_index,:);
f = dnew_3(f_index,:);

u_m = mean(m);
u_f = mean(f);

s_m = cov(m)*(length(m_index)-1);  % matlab normalizes covariance with N-1
s_f = cov(f)*(length(f_index)-1);

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
    data = dnew_4(i,:);
    pred(i) = fisher_judge(data,w,w0,policy);
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

figure; 
hold on; 

plot(m(:,1),m(:,2),'k+','LineWidth',1,'MarkerSize',4);
plot(f(:,1),f(:,2),'ko','MarkerFaceColor','y','MarkerSize',4);
legend('Male','Female');

s = [150,150*w(2)/w(1)+140];
e = [200,200*w(2)/w(1)+140];
%line([s(1),e(1)],[s(2),e(2)],'color','b','linewidth',3); 

bound = proj(0.5*(u_m+u_f), [s(1),e(1),s(2),e(2)]);
endp = [bound(1)+w(2)/w(1)*(bound(2)-210),210];

line([bound(1),endp(1)],[bound(2),endp(2)],'color','r','linewidth',2);
axis([145,210,135,210]); 
hold off; 

figure;
hold on;

plot(d_4(79:250,5),d_4(79:250,6),'k+','LineWidth',1,'MarkerSize',4);
plot(d_4(1:78,5),d_4(1:78,6),'ko','MarkerFaceColor','y','MarkerSize',4);
legend('Male','Female');
line([bound(1),endp(1)],[bound(2),endp(2)],'color','r','linewidth',2);
axis([155,190,135,210]); 
hold off; 