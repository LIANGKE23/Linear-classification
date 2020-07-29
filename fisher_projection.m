function [ W ] = fisher_projection(featureVector,labels,A,B)
%Number of Classes
numofclasses = length(countcats(labels));
%Number of samples points
Samplepoints = size(featureVector,1);
%Number of feature
D = size(featureVector,2);
%mark which position belongs to i class
Mark = double(zeros(Samplepoints,numofclasses));
%count how many number in each class, Ni
Ni = double(zeros(numofclasses,1));
%Get the name of the classes
Labels = unique(labels);
for i=1:numofclasses
    for n = 1:Samplepoints
        if Labels(i) == labels(n)
            Mark(n,i) = 1;
        end
    end
    Ni(i) = sum(double(Mark(:,i)));
end
%get M and mi
mi = double(zeros(numofclasses,D));
for i = 1:numofclasses
    mi(i,:) = sum(double(featureVector(logical(Mark(:,i)),:)))/Ni(i);
end
M = double(zeros(1,D));
for n = 1:D
    M(1,n) = mean(double(mi(:,n)));
end
%get Sw
Sw = double(zeros(D,D));
for i = 1:numofclasses
    for n = 1:Samplepoints
        if Mark(n,i) == 1
            Sw = Sw +(featureVector(n,:)-mi(i,:))'*(featureVector(n,:)-mi(i,:));
        end
    end
end
%get Sb
Sb = zeros(D,D);
for k=1:numofclasses 
    Sb = Sb + Ni(k)*(mi(k,:)-M)'*(mi(k,:)-M);
end
[V,D] = eig(Sw\Sb);
if (A == 1)&&(B == 2)
    W=double(V(:,1:numofclasses));
else
    W=double(V(:,1:(numofclasses-1)));
end