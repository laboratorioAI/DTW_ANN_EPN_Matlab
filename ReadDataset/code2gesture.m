function gesture = code2gesture( code )

    
    switch code
        case 1
            gesture = "noGesture";
        case 2
            gesture = "fist";
        case 3
            gesture = "waveIn";
        case 4
            gesture = "waveOut";
        case 5
            gesture = "open";
        case 6
            gesture = "pinch";
    end
    
end