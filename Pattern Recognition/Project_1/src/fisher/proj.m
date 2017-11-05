function point = proj(p,line_p)
x1 = [line_p(1),line_p(3)];
x2 = [line_p(2),line_p(4)];
x3 = [p(1),p(2)];

k = (x3 - x1) * (x2 - x1)' / sum((x2 - x1).^2);
point = k*(x2 - x1) + x1;
end