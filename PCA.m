clear
close all
%========对图像进行PCA处理并得到特征脸以及匹配结果
% 对样本库的样本进行训练
[W, imgmean, col_of_data, reference] = Training();
[filename, pathname] = uigetfile({'*.bmp'},'choose photo');
str = [pathname, filename];
im = imread(str);
figure
subplot(1,3,1)
imshow(im)
title('原始图像')


% 预处理新数据
im = double(im(:));
objectone = W'*(im - imgmean);
distance = 100000000;

% 最小距离法，寻找和待识别图片最为接近的训练图片
img_path_list = dir(strcat('D:\Project\PCA_face\train','\*.bmp'));

aimpath = 0;
for k = 1:col_of_data
    temp = norm(objectone - reference(:,k));
    if(distance>temp)
        aimone = k;
        distance = temp;
        aimpath = strcat('D:\Project\PCA_face\train', '/', img_path_list(aimone).name);
    end
end

subplot(1,3,2)
imshow(aimpath)
title('匹配图像')
subplot(1,3,3)
imshow(reshape(W(:,aimone),128,128),[])
title('特征脸')













