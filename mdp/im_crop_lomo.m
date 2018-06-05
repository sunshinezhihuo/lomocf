function I = im_crop_lomo(img, bb)

w = bb_width(bb);
h = bb_height(bb);


x1 = max([1 bb(1)]);
y1 = max([1 bb(2)]);
x2 = min([size(img,2) bb(3)]);
y2 = min([size(img,1) bb(4)]);

% patch = img(y1:y2, x1:x2);    % gray
patch = img(y1:y2, x1:x2, :);

x1 = x1-bb(1)+1;
y1 = y1-bb(2)+1;
x2 = x2-bb(1)+1;
y2 = y2-bb(2)+1;

% I(y1:y2, x1:x2) = patch;     %gray
I(y1:y2, x1:x2, :) = patch;

% debug
% figure(100);
% imshow(img);
% rectangle('Position', [bb(1) bb(2) (bb(3)-bb(1)) (bb(4)-bb(2))], 'EdgeColor', 'g', 'LineWidth', 2, 'LineStyle', '-');
% figure(101);
% imshow(I);