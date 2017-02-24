function [img, num] = load_images( list_name, load_size, img_mean )

resize_flag = 0;
mean_flag = 0;

if nargin >= 2
    resize_flag = 1;
end
if nargin >= 3
    mean_flag = 1;
end

f = fopen(list_name);
item_list = {};
idx = 1;
while ~feof(f)
    item_name = fgetl(f);
    item_list{idx} = item_name;
    
    pic = imread(item_name);
    if mean_flag == 1
        pic = single(pic(:,:,1));
    else
        pic = im2single(pic(:,:,1));
    end

    if resize_flag ==1
        pic = imresize(pic, load_size,'bilinear');
    end
    if mean_flag == 1
        pic = pic-img_mean;
        img{idx} = cat(3, pic, pic, pic);
    else 
        img{idx} = pic;
    end
    idx = idx+1;
end
num = idx-1;