% todo: show them on original image

tri = load('tri_refine.mat');
tri = tri.tri + 1;

d = 'obama/';
% wd = 'obama_res@sparse/'; mkdir(wd);
wd = 'obama_res@dense/'; mkdir(wd);
% wd = 'obama_res@point_cloud/'; mkdir(wd);

r = dir([d '*.jpg']);
for ind = 1 : length(r)
    fn = r(ind).name;
    img_fp = [d, fn];
    vertex_fp = strrep(img_fp, '.jpg', '_0.mat');
    
    vertex = load(vertex_fp);
    vertex = vertex.vertex'; % m x 3
    vertex(:, 1) = vertex(:, 1) + 1;
    vertex(:, 2) = vertex(:, 2) + 1;
    vertex(:, 3) = vertex(:, 3) - min(vertex(:, 3));

    img = imread(img_fp);
    gcf = figure('visible', 'off');

    [h, w, c] = size(img);
    set(gcf, 'Position', [0, 0, w, h]);
    imshow(img, 'border', 'tight', 'initialmagnification', 'fit');

    hold on;

    % 1. point cloud
    %pcshow(vertex);

    % 2. mesh
    color = zeros(size(vertex)); % m x 3
    color(:, 1) = 193/255;
    color(:, 2) = 255/255;
    color(:, 3) = 193/255;
    patch('Vertices', vertex, 'Faces', tri, 'FaceVertexCData', color, 'FaceColor', 'interp', 'EdgeColor', 'none', 'EdgeLighting', 'none', 'LineWidth', 1);
    
    % Setting light & position, material, view angle
    camlight('headlight');
    lighting gouraud
%     material dull
    view([0 90]);
    
    % 3. 68 landmarks
%     pts = importdata(pts_fp);
%     pts = pts;
%     plot_landmarks(pts);
    
    
    F = getframe(gcf);
    img = F.cdata;

    % figure; imshow(img);
    img_wfp = [wd, fn];
    disp(img_wfp);
    imwrite(img, img_wfp, 'quality', 96);
end

function [] = plot_landmarks(pts)
    plot(pts(1, :), pts(2, :), '.', 'MarkerSize', 12, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'w');
    
    line_color = 'w';
    plot(pts(1, 1:17), pts(2, 1:17), '-', 'LineWidth', 1, 'Color', line_color);
    plot(pts(1, 18:22), pts(2, 18:22), '-', 'LineWidth', 1, 'Color', line_color);
    plot(pts(1, 23:27), pts(2, 23:27), '-', 'LineWidth', 1, 'Color', line_color);
    plot(pts(1, 28:31), pts(2, 28:31), '-', 'LineWidth', 1, 'Color', line_color);
    plot(pts(1, 32:36), pts(2, 32:36), '-', 'LineWidth', 1, 'Color', line_color);
    
    plot(pts(1, 37:42), pts(2, 37:42), '-', 'LineWidth', 1, 'Color', line_color);
    plot([pts(1, 42), pts(1, 37)], [pts(2, 42), pts(2, 37)], '-', 'LineWidth', 1, 'Color', line_color);
    
    plot(pts(1, 43:48), pts(2, 43:48), '-', 'LineWidth', 1, 'Color', line_color);
    plot([pts(1, 43), pts(1, 48)], [pts(2, 43), pts(2, 48)], '-', 'LineWidth', 1, 'Color', line_color);
    
    plot(pts(1, 49:60), pts(2, 49:60), '-', 'LineWidth', 1, 'Color', line_color);
    plot([pts(1, 49), pts(1, 60)], [pts(2, 49), pts(2, 60)], '-', 'LineWidth', 1, 'Color', line_color);
    
    plot(pts(1, 61:68), pts(2, 61:68), '-', 'LineWidth', 1, 'Color', line_color);
    plot([pts(1, 61), pts(1, 68)], [pts(2, 61), pts(2, 68)], '-', 'LineWidth', 1, 'Color', line_color);
end

