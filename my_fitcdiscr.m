function [ Wstar ] = my_fitcdiscr( featureVector,labels )
%%get the X for training
X = double([ones(size(featureVector,1),1) featureVector]);
%%initialization of T
T = double(zeros(size(featureVector,1),length(countcats(labels))));
for k = 1:size(featureVector,1)
    T(k, labels(k)) = 1;
end
Wstar = (X' * X)\(X' * T);
end

