function[Acc, k] = crossValidation(dataSetNr,nFold,K)
%% Function crossValidation to perform n fold cross validation
%% Input Parameters:
    % dataSetNr: data set number to pick the data from input file. Ex: 1
    % nFold: Number of folds. Ex: 5
    % K: Total number of K neighbours to train the data. Ex: 20
%% Output Parameters:
    % Acc: Maximum Accuracy obtained.
    % k: K value to which maximum accuracy is obtained.
    
%% Reading data
% X - Data samples
% D - Desired output from classifier for each sample
% L - Labels for each sample
[X, D, L] = loadDataSet( dataSetNr );

%% Select a subset of the training samples

numBins = nFold;                    % Number of bins you want to devide your data into
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select samples at random, false = select the first features

[XBins, ~, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom);
%% Look Up matrix to perform cross Validation
mCV = zeros(nFold, nFold); % Initializing Cross Validation matrix with the Number of folds.

for i = 1:nFold
    
    mCV(i,:) = 1:nFold; % Assigning folds indices to the matrix
    
end

mCV = mCV - diag(diag(mCV)); % Making the diagonal bin as validation (by making it zero)
mCV = reshape(nonzeros(mCV'),size(mCV,2)-1,[])'; % removing zeros from matrix
Acc = zeros(1,K); % Initializing Accuracy vector to hold the values of avg accuracy for different K values

for i = 1:K
    trainingAcc = 0;
    for n = 1:nFold
        
        % Combining data from different bins except the bins in diagonals
        XTrain = combineBins(XBins,mCV(n,:));
        LTrain = combineBins(LBins,mCV(n,:));
        
        % Classify the training data
        LTrainPred = kNN(XBins{n},i,XTrain,LTrain);
        
        % The confucionMatrix
        cM = calcConfusionMatrix(LTrainPred, LBins{n});

        % The accuracy
        acc = calcAccuracy(cM);
        
        % Summing up accuracies of n folds for every K value
        trainingAcc = trainingAcc + acc;          
        
    end
   
    Acc(i) = trainingAcc;
    
end

% Normalising the Accuracies by taking average
Acc = Acc./nFold;

[~,k] = max(Acc);

end



