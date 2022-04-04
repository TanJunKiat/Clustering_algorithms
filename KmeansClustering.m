close all
clear all
clc

%% Setting up 
% Define image location
digitDatasetPath = fullfile(pwd);

% Read inidividual file from a folder
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%% Initialisation
img = readimage(imds,5);
img = double(img(:,:,1));
[h w d] = size(img);

% Number of Neurons
Neuron = 4;

%% Counting number of data points
[xi,xj] = find(img == 0);
k = max(size(xi));
figure (1)
scatter (xi,xj,'.')
hold on 

%% Assigning cluster centre
xic = xi(randi([1 k],Neuron,1));
xjc = xj(randi([1 k],Neuron,1));
scatter (xic,xjc,'filled')
hold off
pause (1)

%%
totalsteps = 20;

for steps = 1:totalsteps
% Assigning data to cluster
xihold = zeros(k,Neuron);
xjhold = zeros(k,Neuron);
for i = 1:k
    % Calculating distance of sample to centres
    d = (xi(i)-xic).^2 +(xj(i)-xjc).^2;
    % Finding the closest centre
    [m I] = min(d);
    % Assigning sample to cluster
    xihold(i,I) = xi(i);
    xjhold(i,I) = xj(i);
end

%% Calculating new cluster centres
for i = 1:Neuron
    xic(i) = sum(xihold(:,i))/numel(find(xihold(:,i)));
    xjc(i) = sum(xjhold(:,i))/numel(find(xihold(:,i)));
end

figure (1)
scatter (xic,xjc,'o','filled')
hold on
scatter (xihold,xjhold,'.')
hold off
pause(0.1)
end
