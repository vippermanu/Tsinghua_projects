% input: row vector m containing the data that we want to recognize
%        row vector u containing the mean value of a class's data
%        variance matrix cov of a class
%        dimension d of the data
%        prior probability of this class
% output: posterior probability of this class, given data
function g = bayes_judge(m,u,cov,d,pri)
p = exp(-0.5*(m-u)/cov*(m-u)')/(power(2*pi,d/2)*sqrt(det(cov)));
g = log(p*pri);
end