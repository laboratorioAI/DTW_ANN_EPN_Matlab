function EMGsegment_out = preProcessEMGSegment(EMGsegment_in, Fa, Fb, rectFcn)
% This function pre-process an EMG signal by applying normalization to the 
% range [-1, +] if needed, rectification, and low-pass filtering


    if max( abs(EMGsegment_in(:)) ) > 1
        drawnow;
        EMGnormalized = EMGsegment_in/128;
    else
        EMGnormalized = EMGsegment_in;
    end
    EMGrectified = rectifyEMG(EMGnormalized, rectFcn);
    EMGsegment_out = filtfilt(Fb, Fa, EMGrectified);
    
    
    
end