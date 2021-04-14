function[Acc, k] = crossValidation(dataSetNr,nFold,K)
%% Function crossValidation to perform n fold cross validation
%% Input Parameters:
    % dataSetNr: data set number to pick the data from input file. Ex: 1
    % nFold: Number of folds. Ex: 5
    % K: Total number of K neighbours to train the data. Ex: 20
%% Output Parameters:
    % Acc: Maximum Accuracy obtained.
    % k: K value to which maximum accuracy is obtained.
rng(12345);    
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
AccTrain = zeros(1,K);

for i = 1:K
    trainingAcc = 0;
    testAcc = 0;
    for n = 1:nFold
        
        % Combining data from different bins except the bins in diagonals
        XTrain = combineBins(XBins,mCV(n,:));
        LTrain = combineBins(LBins,mCV(n,:));
        
        % Classify the training data
        LTrainPred = kNN(XTrain,i,XTrain,LTrain);
        LTestPred = kNN(XBins{n},i,XTrain,LTrain);
        
        % The confucionMatrix
        cMTrain = calcConfusionMatrix(LTrainPred, LTrain);
        cMTest = calcConfusionMatrix(LTestPred, LBins{n});

        % The accuracy
        accTrain = calcAccuracy(cMTrain);
        accTest = calcAccuracy(cMTest);
        
        % Summing up accuracies of n folds for every K value
        trainingAcc = trainingAcc + accTrain; 
        testAcc = testAcc + accTest; 
        
    end
   
    Acc(i) = testAcc;
    AccTrain(i)=trainingAcc;
    
end

% Normalising the Accuracies by taking average
Acc = Acc./nFold;

AccTrain = AccTrain./nFold;

% plot
figure; hold on
a1 = plot(1:K,Acc); M1 = "Test Accuracy";
a2 = plot(1:K,AccTrain); M2 = "Training Accuracy";
legend([a1,a2], [M1, M2]);

Acc(1)=0;

disp(Acc);
[k] = max(find(Acc == max(Acc,[],'all')));

end



