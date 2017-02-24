function  feature_vggf  = cnn_feature( img_mb, img_mm, net_vggf, thelayer )


filename = strcat('feature_vggf_l', num2str(thelayer), '.mat');
if ~exist(filename)
    for p = 1:length(img_mb)
        res = vl_simplenn(net_vggf, img_mb{p});
        feature_mb(p, :) = reshape(res(thelayer).x, 1, []);
    end
    for p = 1:length(img_mm)
        res = vl_simplenn(net_vggf, img_mm{p});
        feature_mm(p, :) = reshape(res(thelayer).x, 1, []);
    end
    feature_vggf = [feature_mb;feature_mm];
    eval(char(strcat('save', {' '}, filename, {' '}, 'feature_vggf')))
else
    load(filename);
end

end

