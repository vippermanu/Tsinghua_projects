% input: data matrix x
% output: row vector m, f containing the 
% index of male/female data 
function [m,f] = findindex(x)
m = [];
f = [];
for i = 1:length(x)
    if x{i} == 'M'
        m = [m,i];
    else
        f = [f,i];
    end
end
end