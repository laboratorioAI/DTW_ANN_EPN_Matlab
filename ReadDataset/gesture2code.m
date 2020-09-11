function code = gesture2code(gesture)

    switch gesture
       
        case 'waveOut'
            code = 4;
        case 'waveIn'
            code = 3;    
        case 'fist'
            code = 2;    
        case 'open'
            code = 5;
        case 'pinch'
            code = 6;
        case 'noGesture'
            code = 1; 
            
    end
end
