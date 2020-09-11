function numericalGradient = computeNumericalGradient(costFunction, theta)
% numericalGradient = computeNumericalGradient(costFunction, theta) computes the numerical
% gradient of the function costFunction at the values of theta
% Inputs:
%
% costFunction: cost function handle
% theta: vector of parameters
%
% Output
% numericalGradient: vector with the numerical gradients of the function costFunction
% at the values of theta
%
% Marco E. Benalcázar
% Escuela Politecnica Nacional
% marco.benalcazar@epn.edu.ec
% (C) Copyright Marco E. Benalcázar
%

numericalGradient = zeros(size(theta));
perturbation = zeros(size(theta));
epsilonValue = 1e-4;
for i = 1:numel(theta)
    % Printing the iteration number
    disp(['Iteration: ' num2str(i) ' / ' num2str(numel(theta))]);
    % Setting perturbation vector to epsilon in the component where we want
    % to compute the gradient
    perturbation(i) = epsilonValue;
    % Computing numerical gradient
    cost1 = costFunction(theta + perturbation);
    cost2 = costFunction(theta - perturbation);
    numericalGradient(i) = (cost1 - cost2)/(2*epsilonValue);
    % Setting perturbation vector to 0s
    perturbation(i) = 0;
end   
return