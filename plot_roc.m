clear all;
clc

% Load
net_vggf = load('imagenet-matconvnet-vgg-f.mat');
load('img_mean.mat');
load_size = net_vggf.meta.normalization.imageSize(1:2);
[img3_mb, num_mb] = load_images( 'mb_list', load_size, img_mean );
[img3_mm, num_mm] = load_images( 'mm_list', load_size, img_mean );
[img_mb, num_mb] = load_images( 'mb_list' );
[img_mm, num_mm] = load_images( 'mm_list' );
label_svm = logical([zeros(num_mb, 1); ones(num_mm, 1)]);


% Load HOG
cellnum = 4;
blknum = 4;
hogargs = [cellnum, blknum];
% Load LBP
cellnum = 1;
neighbor = 24;
r = 3;
lbpargs = [cellnum, neighbor, r];
% Load SIFT
class_num = 4;
siftargs = class_num;
% Load CNN
pca_num = 1000;

thelayer = 14;
feature_vggf14 = cnn_feature( img3_mb, img3_mm, net_vggf, thelayer );
% [~, vgg14_pca, ~] = pca(feature_vggf14, 'NumComponents',pca_num);%, 'Economy', false);
load('diff_idx_vggf_l14.mat')
feature_vggf14 = feature_vggf14(:, diff_idx(1:pca_num));

thelayer = 16;
feature_vggf16 = cnn_feature( img3_mb, img3_mm, net_vggf, thelayer );
vgg16_pca_mtx = pca(feature_vggf16, 'Economy', false);
vgg16_pca = (feature_vggf16 * vgg16_pca_mtx);
load('diff_idx_vggf_l16.mat')
feature_vggf16 = feature_vggf16(:, diff_idx(1:pca_num));

thelayer = 18;
feature_vggf18 = cnn_feature( img3_mb, img3_mm, net_vggf, thelayer );
vgg18_pca_mtx = pca(feature_vggf18);%, 'Economy', false);
vgg18_pca = (feature_vggf18 * vgg18_pca_mtx);
load('diff_idx_vggf_l18.mat')
feature_vggf18 = feature_vggf18(:, diff_idx(1:pca_num));

[feature_hog, feature_lbp, feature_sift] = af_feature( img_mb, img_mm, hogargs, lbpargs, siftargs );
feature_all_14 = [feature_hog, feature_lbp, feature_sift, feature_vggf14];
feature_all_16 = [feature_hog, feature_lbp, feature_sift, feature_vggf16];
feature_all_18 = [feature_hog, feature_lbp, feature_sift, feature_vggf18];
% all_pca_mtx = pca(feature_all, 'Economy', false);
% feature_all_pca = (feature_all * all_pca_mtx);
% feature_all_pca = feature2pca(:, 1:pca_num);


%% ROC curve
% Qb = 651; Qm = 386;
% Q1b = floor(Qb*0.90); Q2b = Qb-Q1b;
% Q1m = floor(Qm*0.90); Q2m = Qm-Q1m;
% indb = randperm(Qb);
% indm = randperm(Qm);
% ind1 = [indb(1:Q1b), Qb+indm(1:Q1m)];
% ind2 = [indb(1:Q2b), Qb+indm(1:Q2m)];
% t1 = label_svm(ind1,:);
% t2 = label_svm(ind2,:);

Q = 1037;
Q1 = floor(Q*0.90);
Q2 = Q-Q1;
ind = randperm(Q);
ind1 = ind(1:Q1);
ind2 = ind(Q1+(1:Q2));
t1 = logical(label_svm(ind1,:)-1);
t2 = logical(label_svm(ind2,:)-1);

num=9;
AUCsvm=zeros(num+3,1);

for cir = 1:num
    switch cir
        case 1
            feature = feature_hog;
        case 2
            feature = feature_lbp;
        case 3
            feature = feature_sift;
        case 4
            feature = feature_vggf14;
        case 5
            feature = feature_vggf16;
        case 6
            feature = feature_vggf18;
        case 7
            feature = feature_all_14;
        case 8
            feature = feature_all_16;
        case 9
            feature = feature_all_18;
    end  
    x1 = feature(ind1, :);
    x2 = feature(ind2, :);
    mdlSVM = fitcsvm(x1, t1,  'Standardize',true,...
            'KernelFunction','rbf', 'KernelScale','auto');
    mdlSVM = fitPosterior(mdlSVM);
    [~,score_svm] = predict(mdlSVM, x2);
    score_data{cir} = score_svm;
    [Xsvm,Ysvm,Tsvm,AUCsvm(cir)] = perfcurve(t2,score_svm(:,mdlSVM.ClassNames),'true');
    figure(2);
    switch cir
        case 1
            plot(Xsvm, Ysvm, 'Color', 'b')
        case 2
            plot(Xsvm, Ysvm, 'Color', 'k')
        case 3
            plot(Xsvm, Ysvm, 'Color', 'g')
        case 4
            plot(Xsvm, Ysvm, 'Color', 'r')
        case 5
            plot(Xsvm, Ysvm, 'Color', 'm')
        case 6
            plot(Xsvm, Ysvm, 'Color', 'c')
        case 7
            plot(Xsvm, Ysvm, 'Color', [0.5 0 0.5])
        case 8
            plot(Xsvm, Ysvm, 'Color', [0 0.5 0.5])
        case 9
            plot(Xsvm, Ysvm, 'Color', [0.5 0.5 0])
    end  
    
    hold on;
end

select = [1, 2, 3, 4];
score_svm = 0;
for p = select
    score_svm = score_svm+score_data{p};
end
score_svm = score_svm/length(select);
[Xsvm,Ysvm,Tsvm,AUCsvm(10)] = perfcurve(t2,score_svm(:,mdlSVM.ClassNames),'true');
plot(Xsvm, Ysvm, 'Color', [0.25 0.5 0.75])
hold on;

select = [1, 2, 3, 5];
score_svm = 0;
for p = select
    score_svm = score_svm+score_data{p};
end
score_svm = score_svm/length(select);
[Xsvm,Ysvm,Tsvm,AUCsvm(11)] = perfcurve(t2,score_svm(:,mdlSVM.ClassNames),'true');
plot(Xsvm, Ysvm, 'Color', [0.5 0.75 0.25])
hold on;

select = [1, 2, 3, 6];
score_svm = 0;
for p = select
    score_svm = score_svm+score_data{p};
end
score_svm = score_svm/length(select);
[Xsvm,Ysvm,Tsvm,AUCsvm(12)] = perfcurve(t2,score_svm(:,mdlSVM.ClassNames),'true');
plot(Xsvm, Ysvm, 'Color', [0.75 0.25 0.5])
hold on;
figure(2);
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification by SVM')
legend('hog','lbp','sift','l14','fc1','fc2','l14+','fc1+','fc2+','votel14+','votefc1+','votefc2+')
