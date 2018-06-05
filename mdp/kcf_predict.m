function [ result,tracker, appsim ] = kcf_predict( I, bb,tracker)
%I is the gray image and bb = [x1,y1,x2,y2]
peak_values = zeros(3,1);
scales = zeros(3,1);
results_delta = zeros(3,2);

target_sz = [bb(4)-bb(2),bb(3)-bb(1)];
pos = [bb(2), bb(1)] + target_sz/2;

cont = 1;
for scale_adjust = 0.95:0.05:1.05
%for scale_adjust = 1
    window_sz = floor(tracker.window_sz*tracker.scale*scale_adjust);
    %obtain a subwindow for detection at the position from last
    %frame, and convert to Fourier domain (its size is unchanged)
    patch = get_subwindow_kcf(I, pos, window_sz);
    if(size(patch,1)~=tracker.window_sz(1)||size(patch,2)~=tracker.window_sz(2))
        patch = imResample(patch, tracker.window_sz, 'bilinear');
    end
    zf = fft2(get_features_kcf(patch, tracker.features, tracker.cell_size, tracker.cos_window));
    kzf = gaussian_correlation(zf, tracker.model_xf, tracker.kernel.sigma);
    response = real(ifft2(tracker.model_alphaf .* kzf));  %equation for fast detection
    
    [vert_delta, horiz_delta] = find(response == max(response(:)), 1);
    if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
        vert_delta = vert_delta - size(zf,1);
    end
    if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
        horiz_delta = horiz_delta - size(zf,2);
    end
    
    peak_value = max(response(:));
    if(scale_adjust~=1)
        peak_value = peak_value * 0.95;
    end
    peak_values(cont)=peak_value;
    scales(cont)= tracker.scale*scale_adjust;
    results_delta(cont,:) = [vert_delta-1,horiz_delta-1];
    cont = cont+1;
end
index = find(peak_values == max(peak_values));
tracker.scale = scales(index);
pos = pos + tracker.cell_size * results_delta(index,:); 
target_sz = tracker.init_target_sz * tracker.scale;
x1 = pos(2)-target_sz(2)/2;
y1 = pos(1)-target_sz(1)/2;
x2 = pos(2)+target_sz(2)/2;
y2 = pos(1)+target_sz(1)/2;

result = [x1;y1;x2;y2];

appsim = max(peak_values); 

% debug
% fprintf('kcf value:%f\n',peak_values(index));
% figure(2)
% imshow(I);
% rectangle('Position', [x1 y1 (x2-x1) (y2-y1)], 'EdgeColor', 'r', 'LineWidth', 2, 'LineStyle', '-');
% t = text(30, 30, num2str(appsim),'color','yellow');
% t(1).FontSize = 16;
% hold off;
% figure(3)
% mesh(fftshift(response));

% fprintf('predict scale is %d\n',tracker.scale);
end

