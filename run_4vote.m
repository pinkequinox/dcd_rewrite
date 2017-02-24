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
label_svm = [zeros(num_mb, 1); ones(num_mm, 1)];


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
% [~, vgg16_pca, ~] = pca(feature_vggf16, 'NumComponents',pca_num);%, 'Economy', false);
load('diff_idx_vggf_l16.mat')
feature_vggf16 = feature_vggf16(:, diff_idx(1:pca_num));

thelayer = 18;
feature_vggf18 = cnn_feature( img3_mb, img3_mm, net_vggf, thelayer );
% [~, vgg18_pca, ~] = pca(feature_vggf18, 'NumComponents',pca_num);%, 'Economy', false);
load('diff_idx_vggf_l18.mat')
feature_vggf18 = feature_vggf18(:, diff_idx(1:pca_num));


[feature_hog, feature_lbp, feature_sift] = af_feature( img_mb, img_mm, hogargs, lbpargs, siftargs );
feature_all_14 = [feature_hog, feature_lbp, feature_sift, feature_vggf14];
feature_all_16 = [feature_hog, feature_lbp, feature_sift, feature_vggf16];
feature_all_18 = [feature_hog, feature_lbp, feature_sift, feature_vggf18];
% [~, feature_all_pca, ] = pca(feature_all, 'NumComponents',pca_num);%, 'Economy', false);
% feature_all_pca = feature2pca(:, 1:pca_num);

shuffle = 1;
num = 9;
pred = zeros(1037, num);
for ro = 1:10
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

        fb = feature(1:651, :); fm = feature(652:end, :);
        lb = label_svm(1:651); lm = label_svm(652:end);

        if shuffle
            kb=rand(1,651);
            [m, nb] = sort(kb);
            lb = lb(nb,:);
            fb = fb(nb,:);
            km=rand(1,386);
            [m, nm] = sort(km);
            lm = lm(nm,:);
            fm = fm(nm,:);
            shuffle = 0;
        end
        f{1}=[fb(1:66,:);fm(1:38,:)];
        l{1}=[lb(1:66);lm(1:38)];
        f{2}=[fb(67:131,:);fm(39:76,:)];
        l{2}=[lb(67:131);lm(39:76)];
        f{3}=[fb(132:196,:);fm(77:114,:)];
        l{3}=[lb(132:196);lm(77:114)];
        f{4}=[fb(197:261,:);fm(115:152,:)];
        l{4}=[lb(197:261);lm(115:152)];
        for p = 5:10
            f{p}=[fb((p-1)*65+2:p*65+1,:);fm((p-1)*39-3:p*39-4,:)];
            l{p}=[lb((p-1)*65+2:p*65+1);lm((p-1)*39-3:p*39-4)];
        end
        SVMModel = fitcsvm(cat(1,f{2:10}), cat(1,l{2:10}), 'Standardize',true,...
            'KernelFunction','rbf', 'KernelScale','auto');
        a = predict(SVMModel, f{1});
        b = a;

        for q=2:9
            SVMModel = fitcsvm(cat(1,f{1:q-1},f{q+1:10}), cat(1,l{1:q-1},l{q+1:10}), 'Standardize',true,...
            'KernelFunction','rbf', 'KernelScale','auto');
            a = predict(SVMModel, f{q});
            b = [b;a];

        end
        SVMModel = fitcsvm(cat(1,f{1:9}), cat(1,l{1:9}), 'Standardize',true,...
            'KernelFunction','rbf', 'KernelScale','auto');
        a = predict(SVMModel, f{10});
        b = [b;a];

        pred(:, cir) = b;
    end
    select = [1, 2, 3, 4];
    tp = sum(((sum(pred(:,select),2)>=2) == (cat(1,l{:}))) & ((cat(1,l{:}) == 1)));
    tn = sum(((sum(pred(:,select),2)>=2) == (cat(1,l{:}))) & ((cat(1,l{:}) == 0)));
    accu14(ro,:) = (tp+tn)/1037;
    sens14(ro,:) = tp/386;
    spec14(ro,:) = tn/651;
    
    select = [1, 2, 3, 5];
    tp = sum(((sum(pred(:,select),2)>=2) == (cat(1,l{:}))) & ((cat(1,l{:}) == 1)));
    tn = sum(((sum(pred(:,select),2)>=2) == (cat(1,l{:}))) & ((cat(1,l{:}) == 0)));
    accu16(ro,:) = (tp+tn)/1037;
    sens16(ro,:) = tp/386;
    spec16(ro,:) = tn/651;
    
    select = [1, 2, 3, 6];
    tp = sum(((sum(pred(:,select),2)>=2) == (cat(1,l{:}))) & ((cat(1,l{:}) == 1)));
    tn = sum(((sum(pred(:,select),2)>=2) == (cat(1,l{:}))) & ((cat(1,l{:}) == 0)));
    accu18(ro,:) = (tp+tn)/1037;
    sens18(ro,:) = tp/386;
    spec18(ro,:) = tn/651;
    shuffle = 1;
end
accu_vote(1,:) = mean(accu14);
sens_vote(1,:) = mean(sens14);
spec_vote(1,:) = mean(spec14);

accu_vote(2,:) = mean(accu16);
sens_vote(2,:) = mean(sens16);
spec_vote(2,:) = mean(spec16);

accu_vote(3,:) = mean(accu18);
sens_vote(3,:) = mean(sens18);
spec_vote(3,:) = mean(spec18);