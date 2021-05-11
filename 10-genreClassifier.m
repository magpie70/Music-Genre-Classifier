%%  Music Classification with KNN (10 Genres)
clc
clear
%% Reading csv file
A = readmatrix('features_30_sec.csv');
kd = readcell('features_30_sec.csv', 'delimiter', ',','HeaderLines', 1);
%% Splitting dataset into training and testing
training = [];
testing = [];
features = [2,3,4,6,13,14,16];

Xtrain = [];
Ytrain = [];

Xtest = [];
Ytest = [];

for ii=1:100:1000
    
    training = [training; A(ii:ii+69,features)];
    testing = [testing; A(ii+70:ii+99,features)];
    
    Xtrain = [Xtrain; kd(ii:ii+69,features)];
    Ytrain = [Ytrain; kd(ii:ii+69,60)];

    Xtest= [Xtest; kd(ii+70:ii+99,features)];
    Ytest = [Ytest; kd(ii+70:ii+99,60)];
end

%% Code for counting neighbors and calculating euclidian distance manually
% dist = [];
% k=11;
% answer = [];
% for ii = 1:300
%     dist = [];
%     for step = 1:700
% %         s = 0;
% %         for ff = 1:length(features)
% %             s = s + (training(step,ff)-testing(ii,ff))^2;
% %         end
%         
%         dist = [dist; norm(training(step,:)-testing(ii,:))];
%     end
%     sorted_dist = sort(dist);
%     neighbors_dist = sorted_dist(1:k);
%     neighbors = [];
%     for jj=1:k
%         neighbors = [neighbors find(dist==neighbors_dist(jj),1)];
%     end
%     neighbors = ceil(neighbors/70);
%     answer = [answer mode(neighbors)];
% end
% 
% count = 0;
% for ii = 1:300
%    if answer(ii)==ceil(ii/30)
%       count = count + 1; 
%    end
% end
% count/300

%% Creaing KNN classification model
resp = [];
Mdl = fitcknn(training,Ytrain,'NumNeighbors',11);

%% Testing the data
[label,score,cost] = predict(Mdl,testing);

%% Calculating accuracy of model
count = 0;
 for ii = 1:300
    if strcmp(label(ii),Ytest{ii,1})
        count=count+1;
        
    end
 end
resp =strcmp(label,Ytest);
count/300

%% Calculating and plotting AUC
[Xknn,Yknn,Tknn,AUCknn,OPTPROC] = perfcurve(resp,score(:,2),'true');
plot(Xknn,Yknn)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification')
AUCknn
