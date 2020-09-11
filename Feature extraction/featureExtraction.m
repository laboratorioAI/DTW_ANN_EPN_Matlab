function dataX = featureExtraction(timeSeries, centers, options)
% This function computes a feature vector for each element from the set
% timeSeries. The dimension of this feature vector depends on the number of 
% time series of the set centers. The value of the jth feature of the ith
% vector in dataX corresponds to the DTW distance between the signals 
% timeSeries{i} and centers{j}.
%
% Marco E. Benalcázar
% Escuela Politecnica Nacional
% marco.benalcazar@epn.edu.ec
% (C) Copyright Marco E. Benalcázar
%

numTimeSeries = length(timeSeries);
numClusters = length(centers);
w = options.dtwWindow;
dataX = nan(numTimeSeries, numClusters);
for serie_i = 1:numTimeSeries
    timeSerie_i = timeSeries{ serie_i };
    for center_j = 1:numClusters
        center_cluster_j = centers{ center_j };
        distance = dtw_c( timeSerie_i, center_cluster_j, w );
        dataX(serie_i, center_j) = distance;
    end
end
return