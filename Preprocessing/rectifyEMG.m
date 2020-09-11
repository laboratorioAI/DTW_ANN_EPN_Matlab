function rectifiedEMG = rectifyEMG(rawEMG, rectFcn)
% This function rectifies an EMG signal by applying to each of its values
% any of the following operations: absolute value or square value. If the
% EMG does not need to be rectified, then the option "none" needs to be
% selected

switch rectFcn
    case 'square'
        rectifiedEMG = rawEMG.^2;
    case 'abs'
        rectifiedEMG = abs(rawEMG);
    case 'none'
        rectifiedEMG = rawEMG;
    otherwise
        fprintf(['Wrong rectification function. Valid options are square, ',...
            'abs and none']);
end
return