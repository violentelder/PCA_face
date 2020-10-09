% --- ������ѵ������
function [W, imgmean, col_of_data, reference] = Training()
% W             PCA��Э������������������ɵ�ͶӰ�ӿռ�
% imgmean       ����������������ó���ÿһ�еľ�ֵ��ȥ���Ļ���
% col_of_data   ������ͼ����
% reference     ����������������µı�����
global pathname
global img_path_list
% ������ȡָ���ļ����µ�ͼƬ128*128
pathname = 'D:\Project\PCA_face\train';
img_path_list = dir(strcat(pathname,'\*.bmp'));
img_num = length(img_path_list);
imagedata = [];
if img_num >0
    for j = 1:img_num
        img_name = img_path_list(j).name;
        temp = imread(strcat(pathname, '/', img_name));
        %��ͼ���ά����ת��Ϊ��������ͬʱdouble������֮�������
        temp = double(temp(:));
        imagedata = [imagedata, temp];
    end
end
%��ȡimagedata��������������ͼ����
col_of_data = size(imagedata,2);
% �õ�����ÿһ�еľ�ֵ�������һ��������
imgmean = mean(imagedata,2);

% 1.ȥ���Ļ�
for i = 1:col_of_data
    imagedata(:,i) = imagedata(:,i) - imgmean;
end

%��Э�������covMat
covMat = imagedata' * imagedata;
% ����������COEFF
% ����ֵ��latent
% ÿ������ֵռ�ȣ�explained
[COEFF, latent, explained] = pcacov(covMat);
% ѡ�񹹳�95%����������ֵ
i = 1;
proportion = 0;
while(proportion < 95)
    proportion = proportion + explained(i);
    i = i+1;
end

% %�ڶ����������ֵ�����������ķ���
% �����������eiv������ֵeic
% [eiv, eic] = eig(covMat,'nobalance');
% L_eig_vec = [ ];
% for i = 1:size(eiv)
%     if (eic(i,i) > 10000)
%         L_eig_vec = [L_eig_vec, eiv(:,i)];    %ѡȡ����ֵ����1����������
%     end
% end
% Ei_Face = imagedata * L_eig_vec;

% ������
W = imagedata * COEFF;    % N*M��
W = W(:,1:i - 1);           % N*p��

% ѵ����������������µı����� p*M
reference = W'*imagedata;
end
