function [idxStart, idxEnd] = detectMuscleActivity(emg, options)
% This function segments in a EMG the region corresponding to a muscle
% contraction. The indices idxStart and idxEnd correspond to the begining
% and the end of such a region
%
fs = options.fs; % Sampling frequency of the EMG
% Minimum length of the segmented region
minWindowLengthOfSegmentation = options.minWindowLengthOfSegmentation;
plotSignals = options.plotSignals; % Flag to plot signals
numFreqOfSpec = 50; % The actual number of frequencies is numFreqOfSpec/2
hammingWdwLength = 25; % Window length
numSamplesOverlapBetweenWdws = 10; % Overlap between 2 consecutive windows
threshForSumAlongFreqInSpec = options.threshForSumAlongFreqInSpec;
sumEMG = sum(emg, 2); % Computing the sum of envelopes
% Computing the spectrogram of the EMG
[spec, dummy, time, ps] = spectrogram(sumEMG, hammingWdwLength, ...
    numSamplesOverlapBetweenWdws, numFreqOfSpec, fs, 'yaxis');
% Computing the norm of each value of the spectrogram
spec = abs(spec);
% Summing the spectrogram along the frequencies
sumAlongFreq = sum(spec, 1);
% Thresholding the sum sumAlongFreq
greaterThanThresh = [0, sumAlongFreq >= threshForSumAlongFreqInSpec, 0];
diffGreaterThanThresh = abs(diff(greaterThanThresh));
if diffGreaterThanThresh(end) == 1
    diffGreaterThanThresh(end - 1) = 1;
end
diffGreaterThanThresh = diffGreaterThanThresh(1:(end - 1));
idxNonZero = find(diffGreaterThanThresh == 1);
idxOfSamples = floor(time*fs);
numIdxNonZero = length(idxNonZero);
% Finding the indices of the start and the end of a muscle contraction
switch numIdxNonZero
    case 0
        idxStart = 1;
        idxEnd = length(sumEMG);
    case 1
        idxStart = idxOfSamples(idxNonZero);
        idxEnd = length(sumEMG);
    otherwise
        idxStart = idxOfSamples(idxNonZero(1));
        idxEnd = idxOfSamples(idxNonZero(end) - 1);
end
% Adding a head and a tail to the segmentation
numExtraSamples = 25;
idxStart = max(1, idxStart - numExtraSamples);
idxEnd = min(length(sumEMG), idxEnd + numExtraSamples);
if (idxEnd - idxStart) < minWindowLengthOfSegmentation
    idxStart = 1;
    idxEnd = length(sumEMG);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if plotSignals
    figure(1);
    spectrogram(sumEMG, hammingWdwLength, ...
        numSamplesOverlapBetweenWdws, numFreqOfSpec, fs, 'yaxis');
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('Spectrogram');
    drawnow;  
    figure(2);
    subplot(2, 1, 1);
    plot(sumEMG, 'linewidth', 2, 'Color', [0.9 0.7 0.1]);
    title('Sum of Envelopes');
    subplot(2, 1, 2);
    imagesc( sum(spec, 1) );
    colormap jet;
    colorbar;
    axis off;
    title('Sum of the Spectrogram along the frequencies [W]')
    drawnow;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
return