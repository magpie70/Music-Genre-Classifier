%% Music Classification with KNN (5 Genres)
clc
clear
%% Reading csv file
A = readmatrix('features_30_sec.csv');
kd = readcell('features_30_sec.csv', 'delimiter', ',','HeaderLines', 1);

%% Splitting dataset into training and testing
blues = A(1:100,:);
classical = A(101:200,:);
country = A(201:300,:);
disco = A(301:400,:);
hiphop = A(401:500,:);
jazz = A(501:600,:);
metal = A(601:700,:);
pop = A(701:800,:);
reggae =A(801:900,:);
rock = A(901:1000,:);

blues1 = kd(1:100,:);
classical1 = kd(101:200,:);
country1 = kd(201:300,:);
disco1 = kd(301:400,:);
hiphop1 = kd(401:500,:);
jazz1 = kd(501:600,:);
metal1 = kd(601:700,:);
pop1 = kd(701:800,:);
reggae1 =kd(801:900,:);
rock1 = kd(901:1000,:);

B = [blues; classical; country; hiphop; metal];
C =[blues1; classical1; country1; hiphop1; metal1];
training = [];
testing = [];
features = [2,3,4,6,13,14,16];

Xtrain = [];
Ytrain = [];

Xtest = [];
Ytest = [];

for ii=1:100:500
   training = [training; B(ii:ii+69,features)];
   testing = [testing; B(ii+70:ii+99,features)];
   
   Xtrain = [Xtrain; C(ii:ii+69,features)];
   Ytrain = [Ytrain; C(ii:ii+69,60)];

   Xtest = [Xtest; C(ii+70:ii+99,features)];
   Ytest = [Ytest; C(ii+70:ii+99,60)];
end

%% Code for counting neighbors and calculating euclidian distance manually
% dist = [];
% k=8;
% answer = [];
% for ii = 1:150
%     dist = [];
%     for step = 1:350
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
% for ii = 1:150
%    if answer(ii)==ceil(ii/30)
%       count = count + 1; 
%    end
% end
% 
% count/150

%% Creaing KNN classification model
resp = [];
Mdl = fitcknn(training,Ytrain,'NumNeighbors',10);

%% Testing the data
[label,score,cost] = predict(Mdl,testing);

%% Calculating accuracy of model
count = 0;
 for ii = 1:150
    if strcmp(label(ii),Ytest{ii,1})
        count=count+1;
        
    end
 end
resp = strcmp(label,Ytest);
count/150

%% Calculating and plotting AUC
[Xknn,Yknn,Tknn,AUCknn,OPTPROC] = perfcurve(resp,score(:,2),'true');
plot(Xknn,Yknn)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification')
AUCknn

