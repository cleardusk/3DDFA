function [roi_img, roi_box] = ImageROI(img, roi)
 %% ROI image, handles several situations
    [height, width, nChannels] = size(img);
    sx = roi(1);
    sy = roi(2);
    ex = roi(3);
    ey = roi(4);

    temp = zeros(ey - sy + 1, ex - sx + 1, size(img,3));

    if(sx < 1)
        dsx = 1 - sx + 1;
        sx = 1;
    else
        dsx = 1;
    end

    if(ex > width)
        dex = size(temp,2) - (ex - width);
        ex = width;
    else
        dex = size(temp,2);
    end

    if(sy < 1)
        dsy = 1 - sy + 1;
        sy = 1;
    else
        dsy = 1;
    end

    if(ey > height)
        dey = size(temp,1) - (ey - height);
        ey = height;
    else
        dey = size(temp,1);
    end

    roi_box = [sx, sy, ex, ey];
    temp(dsy:dey, dsx:dex, :) = img(sy:ey, sx:ex, :);
    roi_img = temp;

end