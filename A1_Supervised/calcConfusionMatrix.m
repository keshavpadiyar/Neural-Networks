function [ cM ] = calcConfusionMatrix( LPred, LTrue )
% CALCCONFUSIONMATRIX returns the confusion matrix of the predicted labels

classes  = unique(LTrue);
NClasses = length(classes);

% Add your own code here
cM = zeros(NClasses);

% Build
for i = 1 : size(LPred,1)
    cM( LPred(i), LTrue(i) ) = cM( LPred(i), LTrue(i) ) + 1;
end
end