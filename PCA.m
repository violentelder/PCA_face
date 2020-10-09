clear
close all
%========��ͼ�����PCA�����õ��������Լ�ƥ����
% �����������������ѵ��
[W, imgmean, col_of_data, reference] = Training();
[filename, pathname] = uigetfile({'*.bmp'},'choose photo');
str = [pathname, filename];
im = imread(str);
figure
subplot(1,3,1)
imshow(im)
title('ԭʼͼ��')


% Ԥ����������
im = double(im(:));
objectone = W'*(im - imgmean);
distance = 100000000;

% ��С���뷨��Ѱ�Һʹ�ʶ��ͼƬ��Ϊ�ӽ���ѵ��ͼƬ
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
title('ƥ��ͼ��')
subplot(1,3,3)
imshow(reshape(W(:,aimone),128,128),[])
title('������')













