%starter code for project 2: linear classification
%pattern recognition, CSE583/EE552
%Weina Ge, Aug 2008
%Christopher Funk, Jan 2017
%Bharadwaj Ravichandran, Jan 2020

%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name:Ke LIANG
    PSU Email ID:kul660@psu.edu
    Description: LDA and Fisher-KNN classification.
%}

close all;
clear all;
addpath export_fig
%Choose which problem to solve

Problem = inputdlg('Choose which problem to solve (choices 1 for LDLS, 2 for Fisher):');
P = str2num(cell2mat(Problem));
% P=2;

% Choose which dataset to use (choices wine, wallpaper, taiji) :

DATASET = inputdlg('Choose which dataset to use (choices 1 for wine, 2 for wallpaper, 3 for taiji):');
A = str2num(cell2mat(DATASET));
% A=3;

if A == 1
    dataset = 'wine';
elseif A == 2
    dataset = 'wallpaper';
elseif A == 3
    dataset = 'taiji';
end
[train_featureVector, train_labels, test_featureVector, test_labels] = loadDataset(dataset);
%% An example Linear Discriminant Classification
%  Classification here is based on 2 Features (featureA and feature B).  
%       You will be using all of the features but using 2 features makes it 
%       easier to visualize than the multidimensional hyperplane
if P == 1
    Type = inputdlg('1 for 1-vs-1 or 2 for 1-vs-k:');
    B = str2num(cell2mat(Type));
    if B == 1
        FA = inputdlg('number for feature A:');
        FB = inputdlg('number for feature B:');
        featureA = str2num(cell2mat(FA));
        featureB = str2num(cell2mat(FB));
        feature_idx = [featureA,featureB];
        train_featureVector = double(train_featureVector(:,feature_idx));
        test_featureVector = double(test_featureVector(:,feature_idx));
    elseif B == 2
        feature_idx = 1:size(train_featureVector,2);
        train_featureVector = double(train_featureVector(:,feature_idx));
        test_featureVector = double(test_featureVector(:,feature_idx));
    else
        fprintf("Incorrect input ERROR\n")
    end
elseif P == 2
    Type = inputdlg('1 for 1-vs-1 or 2 for 1-vs-k:');
    B = str2num(cell2mat(Type));
% B=2;
    if B == 1
        FA = inputdlg('number for feature A:');
        FB = inputdlg('number for feature B:');
        featureA = str2num(cell2mat(FA));
        featureB = str2num(cell2mat(FB));
        feature_idx = [featureA,featureB];
%         feature_idx = [1,7];
        train_featureVector = double(train_featureVector(:,feature_idx));
        test_featureVector = double(test_featureVector(:,feature_idx));
    elseif B == 2
        feature_idx = 1:size(train_featureVector,2);
        train_featureVector = double(train_featureVector(:,feature_idx));
        test_featureVector = double(test_featureVector(:,feature_idx));
    else
        fprintf("Incorrect input ERROR\n")
    end   
end
%%  Classify the data and show statistics
if P == 1
    % Train the model (you will have to write this function)
    MdlLinear = my_fitcdiscr(train_featureVector,train_labels);

    % Find the training accurracy (you will have to write testing 
    %      function (the function for finding the class labels from a set of
    %      features)
    train_pred = my_predict(MdlLinear,train_featureVector,train_labels);

    % Find the testing accurracy (you will have to write testing 
    %      function (the function for finding the class labels from a set of
    %      features)
    test_pred = my_predict(MdlLinear,test_featureVector,test_labels);
elseif P == 2
    train_featureVector_projection = train_featureVector * fisher_projection(train_featureVector,train_labels,A,B);
    test_featureVector_projection = test_featureVector * fisher_projection(train_featureVector,train_labels,A,B);
    K = inputdlg('Choose the number of neighbors:');
    k = str2num(cell2mat(K));
%     k=15
    train_pred = classifier_KNN(train_featureVector_projection,train_featureVector_projection,train_labels,k);   
    test_pred = classifier_KNN(train_featureVector_projection,test_featureVector_projection,train_labels,k);       
end
% Create confusion matrix
train_ConfMat = confusionmat(train_labels,train_pred)
% Create classification matrix (rows should sum to 1)
% ASSH = meshgrid(countcats(train_labels));
train_ClassMat = train_ConfMat./(meshgrid(countcats(train_labels))')
% mean group accuracy and std
train_acc = mean(diag(train_ClassMat))
train_std = std(diag(train_ClassMat))
% Create confusion matrix
test_ConfMat = confusionmat(test_labels,test_pred)
% Create classification matrix (rows should sum to 1)
test_ClassMat = test_ConfMat./(meshgrid(countcats(test_labels))')
% mean group accuracy and std
test_acc = mean(diag(test_ClassMat))
test_std = std(diag(test_ClassMat))
%%  Display the linear discriminants and a set of features in two of the feature dimensions
%      You will need to modify this function for to use your LDA
%      classification boundries to work with your code.. Look at the code
%      for more details
if P == 1
    if(A == 1)&&(B == 2)
    else
        figure(1)
        visualizeBoundaries(MdlLinear,test_featureVector,test_labels,1,2)
        title('{\bf Linear Discriminant Classification}')
        export_fig linear_discriminant_example -png -transparent
    end
    %%  Display the classified regions of two of the feature dimensions  
    %      You will need to modify this function for with your testing 
    %      function (the function for finding the class labels from a set of
    %      features).
    if (B == 1)&&(A == 1)
        figure(2)
        h = visualizeBoundariesFill(MdlLinear,train_featureVector,train_labels,test_featureVector,test_labels,1,2,P);
        title('{\bf Classification Area}')
        export_fig classification_fill_example -png -transparent
    end
elseif P == 2
    if (B == 1)&&(A == 1)   
        MdlLinear = k;
        figure(2)
        h = visualizeBoundariesFill(MdlLinear,train_featureVector,train_labels,test_featureVector,test_labels,1,2,P);
        title('{\bf Classification Area}')
        export_fig classification_fill_example -png -transparent
    end
end