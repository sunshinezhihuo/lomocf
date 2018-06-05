function [ tracker ] = kcf_update( I,tracker )
%KCF_UPDATE Summary of this function goes here
%   Detailed explanation goes here


target_sz = [tracker.bb(4)-tracker.bb(2),tracker.bb(3)-tracker.bb(1)];
pos = [tracker.bb(2), tracker.bb(1)] + target_sz/2;
window_sz = floor(target_sz * (1 + tracker.padding));

tracker.scale = target_sz/tracker.init_target_sz;

%obtain a subwindow for detection at the position from last
%frame, and convert to Fourier domain (its size is unchanged)
patch = get_subwindow_kcf(I, pos, window_sz);
if(size(patch,1)~=tracker.window_sz(1)||size(patch,2)~=tracker.window_sz(2))
    patch = imResample(patch, tracker.window_sz, 'bilinear');
end

xf = fft2(get_features_kcf(patch, tracker.features, tracker.cell_size, tracker.cos_window));
kf = gaussian_correlation(xf, xf, tracker.kernel.sigma);
alphaf = tracker.yf ./ (kf + tracker.lambda);
if(tracker.bb_overlaps(tracker.anchor)>tracker.threshold_box)
    tracker.model_alphaf = (1 - tracker.interp_factor_1) * tracker.model_alphaf + tracker.interp_factor_1 * alphaf;
    tracker.model_xf = (1 - tracker.interp_factor_1) * tracker.model_xf + tracker.interp_factor_1 * xf;
else
    tracker.model_alphaf = (1 - tracker.interp_factor_2) * tracker.model_alphaf + tracker.interp_factor_2 * alphaf;
    tracker.model_xf = (1 - tracker.interp_factor_2) * tracker.model_xf + tracker.interp_factor_2 * xf;
end
%fprintf('update scale is %d\n',tracker.scale);

end

