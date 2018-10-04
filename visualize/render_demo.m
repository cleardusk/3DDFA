tri = load('tri.mat');
vertex = load('image00427');

tri = tri.tri;
vertex = vertex.vertex;
render_face_mesh(vertex, tri);

A = getframe(gcf);
mimg = A.cdata;
imwrite(mimg, 'demo.jpg', 'quality', 95);
