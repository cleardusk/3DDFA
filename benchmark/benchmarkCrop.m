std_size = 120;

d = 'AFLW2000-3D';
wd = 'AFLW2000-3D_crop';

fns = dir([d '/' '*.jpg']);
nImages = length(fns);

parfor i = 1 : nImages
    img_fp = [d '/' fns(i).name];
    img_wfp = [wd '/' fns(i).name];
    roi_box_wfp = strrep(img_wfp, '.jpg', '_roi_box.mat');
    img = double(imread(img_fp)) / 255;

    param_fp = strrep(img_fp, '.jpg', '.mat');
    param = load(param_fp);

    % for ALFW-2000D
    pts = param.pt3d_68;

    % Crop face
    roi_box = calcRoiBox(pts);
    [roi_img, roi_box] = ImageROI(img, roi_box);
    roi_img = imresize(roi_img, [std_size, std_size]);
    imwrite(roi_img, img_wfp, 'Quality', 95);

    parsave(roi_box_wfp, roi_box);
end

function [roi_box] = calcRoiBox(pts)
    bbox = [min(pts(1,:)), min(pts(2,:)), max(pts(1,:)), max(pts(2,:))];
    center = [(bbox(1)+bbox(3))/2, (bbox(2)+bbox(4))/2];
    radius = max(bbox(3)-bbox(1), bbox(4)-bbox(2)) / 2;
    bbox = [center(1) - radius, center(2) - radius, center(1) + radius, center(2) + radius];

    llength = sqrt( (bbox(3)-bbox(1)).^2  + (bbox(4)-bbox(2)).^2 );
    center_x = (bbox(3) + bbox(1))/2;
    center_y = (bbox(4) + bbox(2))/2;
    roi_box(1) = round(center_x - llength / 2);
    roi_box(2) = round(center_y - llength / 2);
    roi_box(3) = roi_box(1) + llength;
    roi_box(4) = roi_box(2) + llength;
    roi_box = round(roi_box);
end

function parsave(wfp, para)
    save(wfp, 'para');
end
