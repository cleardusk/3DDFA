model = load('model_refine.mat');
model = model.model_refine;

mu = model.mu_shape + model.mu_exp;
mu = reshape(mu, 3, length(mu) / 3);
tri = model.tri;
keypoints = model.keypoints;
pts68_3d = mu(:, keypoints);

render_face_mesh(mu, tri, pts68_3d);

A = getframe(gcf);
mimg = A.cdata;
imwrite(mimg, 'imgs/bfm_noneck.jpg', 'quality', 95);
