function y = elu(x)
% elu(x) computes the value of the elu transfer function for the elements
% of x
%
% Marco E. Benalcázar
% Escuela Politecnica Nacional
% marco.benalcazar@epn.edu.ec
% (C) Copyright Marco E. Benalcázar
%
y = (x > 0).*x + (x <= 0).*(exp(x) - 1);
return