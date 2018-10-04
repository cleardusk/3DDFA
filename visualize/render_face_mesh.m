function render_face_mesh(vertex, tri)
    trisurf(tri', vertex(1, :), vertex(2, :), vertex(3, :), ones(size(vertex, 2),1), 'edgecolor', 'none');
    
    re=[1 1 1];
    colormap(re);

    light('Position', [0 0 1], 'Style', 'infinite');
    lighting gouraud
    axis equal
    view([0 90]);
    
    xlabel('x');
    ylabel('y');
    zlabel('z');
    
    axis on;
    grid on;
end
