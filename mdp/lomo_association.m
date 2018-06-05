function [ descriptors ] = lomo_association( I,bb,tracker,opt )
%KCF_ASSOCIATION Summary of this function goes here

options.numScales = tracker.numScales;     %..
options.blockSize = tracker.blockSize;    %..
options.blockStep = tracker.blockStep;
options.hsvBins = tracker.hsvBins;
options.tau = tracker.tau;
options.R = tracker.R;        %..
options.numPoints = tracker.numPoints;

% debug
% figure(100);
% imshow(I);
% rectangle('Position', [bb(1) bb(2) (bb(3)-bb(1)) (bb(4)-bb(2))], 'EdgeColor', 'g', 'LineWidth', 2, 'LineStyle', '-');
% hold on;

Icrop = im_crop_lomo(I, bb);     % color

images(:,:,:,1) = Icrop;
descriptors = LOMO(images, options);


end

