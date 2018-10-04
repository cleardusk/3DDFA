tri = load('tri.mat');
vertex = load('image00427');

tri = tri.tri;
vertex = vertex.vertex;
render_face_mesh(vertex, tri);