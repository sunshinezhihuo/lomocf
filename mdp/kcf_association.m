function [ peak_value ] = kcf_association( I,bb,tracker )
%KCF_ASSOCIATION Summary of this function goes here
%   Detailed explanation goes here
flag = 0;
target_sz = [bb(4)-bb(2),bb(3)-bb(1)];
pos = [bb(2), bb(1)] + target_sz/2;
window_sz = floor(target_sz * (1 + tracker.padding));
patch = get_subwindow_kcf(I, pos, window_sz);

if(size(patch,1)~=tracker.window_sz(1)||size(patch,2)~=tracker.window_sz(2))
    patch = imResample(patch, tracker.window_sz, 'bilinear');
end

%debug
% fprintf('the patch size is %d,%d\n',size(patch,1),size(patch,2));
% fprintf('the window size is %d,%d\n',tracker.window_sz(1),tracker.window_sz(2));

zf = fft2(get_features_kcf(patch, tracker.features, tracker.cell_size, tracker.cos_window));
kzf = gaussian_correlation(zf, tracker.model_xf, tracker.kernel.sigma);
response = real(ifft2(tracker.model_alphaf .* kzf));  %equation for fast detection
peak_value = max(response(:));


end

