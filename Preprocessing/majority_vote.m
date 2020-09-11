function class_maj = majority_vote(class,before,after)

class_maj = zeros(size(class)); 

for i = 1:length(class)
    window = class(max(1,(i-before)):min(length(class),(i+after))); 
    for j = 1:max(class)
        votes(j) = sum(window == j);
    end
    class_maj(i) = min(find(votes == max(votes))); 
end
