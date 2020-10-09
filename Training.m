% --- 样本集训练函数
function [W, imgmean, col_of_data, reference] = Training()
% W             PCA后协方差矩阵的特征向量组成的投影子空间
% imgmean       样本集列向量化后得出的每一行的均值（去中心化）
% col_of_data   样本集图像数
% reference     样本集在新坐标基下的表达矩阵
global pathname
global img_path_list
% 批量读取指定文件夹下的图片128*128
pathname = 'D:\Project\PCA_face\train';
img_path_list = dir(strcat(pathname,'\*.bmp'));
img_num = length(img_path_list);
imagedata = [];
if img_num >0
    for j = 1:img_num
        img_name = img_path_list(j).name;
        temp = imread(strcat(pathname, '/', img_name));
        %将图像二维矩阵转换为列向量，同时double化便于之后的运算
        temp = double(temp(:));
        imagedata = [imagedata, temp];
    end
end
%获取imagedata的列数，即样本图像数
col_of_data = size(imagedata,2);
% 得到矩阵每一行的均值，并组成一个列向量
imgmean = mean(imagedata,2);

% 1.去中心化
for i = 1:col_of_data
    imagedata(:,i) = imagedata(:,i) - imgmean;
end

%求协方差矩阵covMat
covMat = imagedata' * imagedata;
% 特征向量：COEFF
% 特征值：latent
% 每个特征值占比：explained
[COEFF, latent, explained] = pcacov(covMat);
% 选择构成95%能量的特征值
i = 1;
proportion = 0;
while(proportion < 95)
    proportion = proportion + explained(i);
    i = i+1;
end

% %第二种求解特征值及特征向量的方法
% 获得特征向量eiv及特征值eic
% [eiv, eic] = eig(covMat,'nobalance');
% L_eig_vec = [ ];
% for i = 1:size(eiv)
%     if (eic(i,i) > 10000)
%         L_eig_vec = [L_eig_vec, eiv(:,i)];    %选取特征值大于1的特征向量
%     end
% end
% Ei_Face = imagedata * L_eig_vec;

% 特征脸
W = imagedata * COEFF;    % N*M阶
W = W(:,1:i - 1);           % N*p阶

% 训练样本在新坐标基下的表达矩阵 p*M
reference = W'*imagedata;
end
