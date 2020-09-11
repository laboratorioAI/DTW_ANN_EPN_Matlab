% Copyright (C) 2013 Quan Wang <wangq10@rpi.edu>,
% Signal Analysis and Machine Perception Laboratory,
% Department of Electrical, Computer, and Systems Engineering,
% Rensselaer Polytechnic Institute, Troy, NY 12180, USA
% 
% This function computes the Dynamic Time Warping distance between two time
% series.
%
% Link from the code: https://la.mathworks.com/matlabcentral/fileexchange/43156-dynamic-time-warping--dtw- 
%
% The following lines of code are a demo for this function
%
% clear;clc;close all;
% 
% mex dtw_c.c;
% 
% a=rand(500,3);
% b=rand(520,3);
% w=50;
% 
% tic;
% d=dtw_c(a,b,w);
% t=toc;
% 
% fprintf('Using C/MEX version: distance=%f, running time=%f\n',d,t);

