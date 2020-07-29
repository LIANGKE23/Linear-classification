function [pred] = classifier_KNN(train_featureVector,test_featureVector,train_labels,K)
for i=1:size(test_featureVector,1)
    New_test = double(repmat(test_featureVector(i,:), size(train_featureVector, 1), 1));
    distance = double(sqrt(sum((train_featureVector - New_test).^2, 2)));
    [x,y] = sort(distance);
    %find the biggest one
    pred(i) = mode(train_labels(y(1:K)))';
end
end  
  