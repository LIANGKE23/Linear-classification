function [ pred ] = my_predict( MdlLinear,featureVector,labels )
%%get the X for training
X = [ones(size(featureVector,1),1) featureVector];
Y = X * MdlLinear;
for i = 1:size(featureVector,1)
    [data, position] = max(Y(i,:));
    Labels = unique(labels);
    pred(i,1) = Labels(position);
end
end

