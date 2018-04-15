clear all;
close all ;
%This code for detail map 
p = 10; % samples
pp =1; %9% view point 
folder_set = {'011', '016', '021', '022', '033', '036', '038', '053', '059', '092'};
aloss_total = 0;
aloss_total2 = 0;
err_total = 0;
err_total2 = 0;
min_aloss = 100.0;
max_aloss = -100.0;
good_pixel1 = 10;
good_pixel2 = 15;
good_pixel3 = 20;
A1_total = 0;
A2_total = 0;
A3_total = 0;

values=[];
objects=[];
view=[];
author = 'yoon';
count =1;
for i = 1: p
    for j=2:2%5:5
        folder = fullfile(folder_set{i});
        type = 'L2ang_ei'; %L1,L2,L1ang,L2ang,L1light,L2ang_ei,eigen,L1ang_ei
        im1 = imread(sprintf('%s%s%d%s%s',folder,'/',j,'/','12_Normal.bmp')); 
        
        im1 = im2double(im1);
        im1 = imresize(im1,0.5);
        
        % Extract only detail map
        G = fspecial('gaussian', [71 71], 5);
        im1_d = imfilter(im1,G,'same');
        
        
        im2 = imread(sprintf('%s%s%d%s%s%s.bmp',folder,'/',j,'/','single_normal_',type));
        
 
        im2 = im2double(im2);
        if strcmp(author,'other')
            im2 = imresize(im2,[600,800],'bicubic');
        end
        
       % Extract only detail map
%         G = fspecial('gaussian', [71 71], 5);
        im2_d = imfilter(im2,G,'same');
        im2 = im2-im2_d;
        
        im2 = im2+im1_d;

        mask = imread(sprintf('%s%s%d%s',folder,'/',j,'/mask.bmp'));
        mask = imresize(mask,0.5);

        n = sum(sum(mask));

        im1 = im1.*2-1;
        im2 = im2.*2-1;

        imshow(im1);

        %%%% Mean Error%%%%%
        im1_ = im1.*repmat(mask,1,1,3);
        im2_ = im2.*repmat(mask,1,1,3);
        err = im1_ - im2_;
        err = abs(err);
        err = err.*double(repmat(mask,1,1,3));
        err_mean = sum(sum(sum(err)));
        err_mean = err_mean./n./3
        
        err_total = err_total+err_mean;
        
        %%%% Median Error%%%%%
        tmp = err(err~=0);
        err_median = median(tmp);
        err_total2 = err_total2 + err_median;
        
        %%%% Angular Error %%%%
        
        r = size(im1,1);    c = size(im1,2);
        im1_ = im1;
        im2_ = im2;
        
        im1_v = reshape(im1_,[r*c,3]);
        im2_v = reshape(im2_,[r*c,3]);
        
        norm1 = sqrt ( im1_v(:,1).^2 + im1_v(:,2).^2+im1_v(:,3).^2 );
        norm2 = sqrt ( im2_v(:,1).^2 + im2_v(:,2).^2+im2_v(:,3).^2 );
        norm1_3 = repmat(norm1,1,3);
        norm2_3 = repmat(norm2,1,3);
        
        im1_vn = im1_v./norm1_3;
        im2_vn = im2_v./norm2_3;
        
        ang = im1_vn'.*im2_vn';
        ang = sum(ang);
        ang_ = reshape(ang',r,c);
        ang_m = ang_.*double(mask);
        %%%% Convert to radian angles
        imshow(ang_m);
        %tmp_ang = 1-ang_m;
        %tmp_ang = sum(tmp_ang(:))/sum(mask(:))
        
        ang_rad = acosd(ang_m);
        se = strel('line',30,400);
        mask = imerode(mask,se);
        ang_rad_m = ang_rad.*double(mask);
        
        %ang_rad_m = ang_rad_m(5:end-5,5:end-5);
        
        imagesc(ang_rad_m);
        colormap jet;
        colorbar;
        % aloss = mean(ang_rad_m(:))
        aloss = sum(ang_rad_m(:))./sum(mask(:))
        aloss_total = aloss_total+aloss;
        tmp2 = ang_rad_m(ang_rad_m~=0);
        aloss_median = median(tmp2);
        aloss_total2 = aloss_total2 + aloss_median;
        
        A1 = length(find(tmp2<good_pixel1)) ./ length(tmp2) * 100;
        A2 = length(find(tmp2<good_pixel2)) ./ length(tmp2) * 100;
        A3 = length(find(tmp2<good_pixel3)) ./ length(tmp2) * 100;
        
        A1_total = A1_total+A1;
        A2_total = A2_total+A2;
        A3_total = A3_total+A3;
        
        if aloss > max_aloss
            max_aloss = aloss;
            max_class = folder_set{i};
            max_tilt = j
        end
        
        if min_aloss > aloss
            min_aloss = aloss;
            min_class =folder_set{i};
            min_tilt = j
        end
        values(count)= aloss;
        objects(count) = i;
        view(count) = j;
        count = count+1;
    end
end


[val,ord] = sort(values, 'descend');
objects = objects(ord);
view = view(ord);

err_total_mean = err_total/(p*pp);
err_total_median = err_total2/(p*pp);
aloss_total_mean = aloss_total/(p*pp);
aloss_total_median = aloss_total2/(p*pp);
A1_total_mean = A1_total/(p*pp);
A2_total_mean = A2_total/(p*pp);
A3_total_mean = A3_total/(p*pp);

disp(['Mean of Absolute Error is ' num2str(err_total_mean)]);
disp(['Median of Absolute Error is ' num2str(err_total_median)]);

disp(['Mean of Angular Error is ' num2str(aloss_total_mean)]);
disp(['Median of Angular Error is ' num2str(aloss_total_median)]);
disp(['Max angular error  ' num2str(max_aloss)]);
disp(['Min angular error  ' num2str(min_aloss)]);

disp(['Ratio within 10 deg ' num2str(A1_total_mean)]);
disp(['Ratio within 15 deg ' num2str(A2_total_mean)]);
disp(['Ratio within 20 deg ' num2str(A3_total_mean)]);



