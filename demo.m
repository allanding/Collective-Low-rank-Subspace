% clc
% clear all
% close all


%% load data for 2 views
warning off
K = 2; %% the number of view/classes/sources
Xs = cell(1,K);
si = cell(1,K);

%% data preparation
load 2view1.mat
Xt1 = NormalizeFea(Xt1);
Xt2 = NormalizeFea(Xt2);
Xs1 = NormalizeFea(Xs1);
Xs2 = NormalizeFea(Xs2);

Xtt = [Xt1;Xt2]';
Ytt = [Yt1;Yt2];
Xss = [Xs1;Xs2]';
Yss = [Ys1;Ys2];


%% Initialize the data and variable matrices
si{1} = size(Xs1,1);
si{2} = size(Xs2,1);

Xs{1} = Xs1';
Xs{2} = Xs2';
Ys{1} = Ys1;
Ys{2} = Ys2;
options.K = K;
options.ReducedDim = 200;
options.lambda3 = 1e1;
options.optP = 2;
options.inner = 100;
Pt = CLRS(Xs,Ys,options);

%% Test Stage
Zs = Pt'*Xss;
Zt = Pt'*Xtt;
Cls = cvKnn(Zt, Zs, Yss, 1);
acc = length(find(Cls==Ytt))/length(Ytt);
fprintf('NN=%0.4f\n',acc);


