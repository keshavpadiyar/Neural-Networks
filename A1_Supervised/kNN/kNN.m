function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)

% Variable Initialization
    data = XTrain;
    x = X;
    y = LTrain;
    [d_nrow, d_ncol] = size(data);
    [x_nrow, x_ncol] = size(x);
    dist = zeros(size(data,1),1);
    LPred  = zeros(size(X,1),1);
    final_class = zeros(size(data,1),1);

% Row wise looping to calculate Eucledian Distance between data points
% Summing the distances obtained from all variables (columns) for a
% particular row
    for x_index = 1:x_nrow
         for d_index = 1:d_nrow
            dist(d_index) = sqrt(sum((data(d_index,:)-x(x_index,:)).^2)); % Eucledian Distance
            final_class(d_index)=y(d_index);
         end
        [d,i] = sort(dist); % sort the distances in ascending order
        i = i(1:k); % select the k nearest distance
        [c,v] = groupcounts(final_class(i)); % get the element wise count (similar to table operation in R)
        v = v(find(c==max(c),1)); % Majority Voting (if many classes distances comes same for a data point then select the first class)
        LPred(x_index) = v;
    end

end % end function LPred

