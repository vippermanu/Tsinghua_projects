% input: row vector data that we want to recognize
%        transformation vector w
%        judge boundary w0
%        bool value policy showing that whether u_m is bigger than w0
% output: 1 for male and 0 for female
function n = fisher_judge(data,w,w0,policy)
y = w' * data';
if policy == 1
    n = y > w0;
else
    n = ~(y>w0);
end
end