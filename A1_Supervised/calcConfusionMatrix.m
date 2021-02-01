function [ cM ] = calcConfusionMatrix( LPred, LTrue )
% CALCCONFUSIONMATRIX returns the confusion matrix of the predicted labels

classes  = unique(LTrue);
NClasses = length(classes);

% Add your own code here
cM = zeros(NClasses);

% One Hot Encoding for classes
oneHotLTrue = zeros(length(LTrue),NClasses);
oneHotLPred = zeros(length(LPred),NClasses);

for i = 1:NClasses
    for j = 1:length(LTrue)
        if LTrue(j)==classes(i)
            oneHotLTrue(j,i)=1;
        end
        if LPred(j)==classes(i)
            oneHotLPred(j,i)=1;
        end
    end
end

% Build

for i = 1:NClasses
    for j = 1:NClasses
        cM(i,j) = sum(oneHotLTrue(:,i)+oneHotLPred(:,j)==2);
    end
end
end