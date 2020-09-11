function code = codeSamples(kRep,gestureData)

    rep = sprintf('idx_%d',kRep);


    if kRep < 26

        code = gesture2code(gestureData.(rep).gestureName);  

    elseif kRep < 51

        code = gesture2code(gestureData.(rep).gestureName); 

    elseif kRep < 76

        code = gesture2code(gestureData.(rep).gestureName); 

    elseif kRep < 101

        code = gesture2code(gestureData.(rep).gestureName); 

    elseif kRep < 126

        code = gesture2code(gestureData.(rep).gestureName); 

    elseif kRep < 151

        code = gesture2code(gestureData.(rep).gestureName); 

    end









end