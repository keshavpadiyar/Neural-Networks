%% Hyper-parameters

% Number of randomized Haar-features
nbrHaarFeatures = 25;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 500;
% Number of weak classifiers
nbrWeakClassifiers = 30;

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

%% Create image sets (do not modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError
T = 200; %Number of Weak Classifiers
D = ones(1,size(xTrain,2))/size(xTrain,2); % initializing initial weights

% Variables to store the optimal outputs of each weak classifier
h_alpha = zeros(T,1);
h_threshold = zeros(T,1);
h_feature = zeros(T,1);
h_polarity = zeros(T,1);

for c = 1:T
    
    %  Variable Initialization
    emin = inf; % Epsilon Loss Term

    for k = 1:size(xTrain,1) %looping over all the features (the input matrix is of the form (feature,observations)
        threshold = xTrain(k,:)+0.02; % initializing thresholds (All the observations in a feature will act as threshold)
        % Added 0.05 offset, to avoid overfitting.

        for t = threshold %looping over all the thresholds and identify the best threshold
            p = 1; % polarity
            h = WeakClassifier(t, p, xTrain(k,:)); % Classification using weak classifier
            e = WeakClassifierError(h, D, yTrain); % Loss function 

            if e>0.5 % for a decision stump if loss is >0.5 we shift the polarity
                e = 1-e;
                p = -p;
            end
            
            % Capture the parameters where loss is the least
            if e<emin
                emin = e;
                alpha = 0.5 * log ((1 - emin)/(emin));
                optimalThreshold = t;
                optimalPolarity = p;
                optimalFeature = k;
                optimalClassification = p*h;
            end
        end
    end
 
    if emin == 0.5 % if loss is 0.5 then alpha = 0 hence weight wont be updated so we break at this stage      
        break;
    end
    
    D = D .* exp(-alpha * yTrain .* optimalClassification); % Updating weights    
    D(D>0.5) = 0.5; % trimming weights to decrease the effects of outliers    
    D = D ./ sum(D); % Normalizing weights
    
    % Updating Variables to store the optimal outputs of each weak classifier
    h_alpha(c)= alpha;
    h_threshold(c) = optimalThreshold;
    h_feature(c) = optimalFeature;
    h_polarity(c) = optimalPolarity;        
end
%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.

h = zeros(length(h_threshold),size(xTest,2));
for i = 1:size(h,1)
    h(i,:)= h_alpha(i) * WeakClassifier(h_threshold(i),h_polarity(i),xTest(h_feature(i),:)); 
end

h_classification = sign(sum(h,1));
Accuracy = sum(h_classification==yTest)/size(yTest,2)
%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.



%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.



%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.


