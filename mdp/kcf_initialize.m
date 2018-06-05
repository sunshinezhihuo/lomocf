function [tracker ] = kcf_initialize( I, bb,tracker,opt,frame_id)
switch opt.method
    case 'KCF'
        %kcf initialize,the input is gray image,target box and coefficency.

        cell_size = tracker.cell_size;
        features = tracker.features;
        kernel = tracker.kernel;
        padding = tracker.padding;
        output_sigma_factor = tracker.output_sigma_factor;
        lambda = tracker.lambda;
        template_sz = tracker.template_sz;
        scale = 1;

        %set initial position and size
        target_sz = [bb(4)-bb(2),bb(3)-bb(1)];
        pos = [bb(2), bb(1)] + target_sz/2;

        %if the target is large, lower the resolution, we don't need that much
        %detail
        if (sqrt(prod(target_sz)) >= template_sz)
            if(target_sz(1)>target_sz(2))
                scale = target_sz(1)/template_sz;
            else
                scale = target_sz(2)/template_sz;
            end
        end
        target_sz = floor(target_sz/scale);

        %window size, taking padding into account
        window_sz = floor(target_sz * (1 + padding));

        %create regression labels, gaussian shaped, with a bandwidth
        %proportional to target size
        output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
        yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));

        %store pre-computed cosine window
        cos_window = hann(size(yf,1)) * hann(size(yf,2))';	

        if size(I,3) > 1,
            I = rgb2gray(I);
        end

        extracted_sz = floor(window_sz * scale);

        %obtain a subwindow for training at newly estimated target position
        patch = get_subwindow_kcf(I, pos, extracted_sz);
        if(size(patch,1)~=window_sz(1)||size(patch,2)~=window_sz(2))
            patch = imResample(patch, window_sz, 'bilinear');
        end
        xf = fft2(get_features_kcf(patch, features, cell_size, cos_window));

        kf = gaussian_correlation(xf, xf, kernel.sigma);
        alphaf = yf ./ (kf + lambda);   %equation for fast training
        model_alphaf = alphaf;
        model_xf = xf;

        tracker.init_target_sz = target_sz;
        tracker.model_alphaf = model_alphaf;
        tracker.model_xf = model_xf;
        tracker.cos_window = cos_window;
        tracker.window_sz = window_sz;
        tracker.yf = yf;
        tracker.scale = scale;
        %fprintf('init scale is %d\n',tracker.scale);
    case 'BACF'
        %%   Setting parameters for local use.
        search_area_scale = tracker.search_area_scale;
        output_sigma_factor = tracker.output_sigma_factor;
        learning_rate = tracker.learning_rate;
        filter_max_area = tracker.filter_max_area;
        nScales = tracker.number_of_scales;
        scale_step = tracker.scale_step;
        template_sz = tracker.template_sz;  % from kcf     
        interpolate_response = tracker.interpolate_response;
        
        features = tracker.features;
        scale = 1;    % me kcf add
        
        %set initial position and size
        target_sz = [bb(4)-bb(2),bb(3)-bb(1)];
        pos = [bb(2), bb(1)] + target_sz/2;
        
        %if the target is large, lower the resolution, we don't need that much
        %detail  %----------
% %         if (sqrt(prod(target_sz)) >= template_sz)
% %             if(target_sz(1)>target_sz(2))
% %                 scale = target_sz(1)/template_sz;
% %             else
% %                 scale = target_sz(2)/template_sz;
% %             end
% %         end
% %         
% %         target_sz = floor(target_sz/scale);
        
        init_target_sz = target_sz;
        
%         %window size, taking padding into account
%         window_sz = floor(target_sz * (1 + padding));  %kcf              
        
        featureRatio = tracker.t_global.cell_size;
        search_area = prod(init_target_sz / featureRatio * search_area_scale);
        
        % when the number of cells are small, choose a smaller cell size
        if isfield(tracker.t_global, 'cell_selection_thresh')
            if search_area < tracker.t_global.cell_selection_thresh * filter_max_area
                tracker.t_global.cell_size = min(featureRatio, max(1, ceil(sqrt(prod(init_target_sz * search_area_scale)/(tracker.t_global.cell_selection_thresh * filter_max_area)))));

                featureRatio = tracker.t_global.cell_size;
                search_area = prod(init_target_sz / featureRatio * search_area_scale);
            end
        end
        global_feat_params = tracker.t_global;
        
        if search_area > filter_max_area
            currentScaleFactor = sqrt(search_area / filter_max_area);
        else
            currentScaleFactor = 1.0;
        end
          currentScaleFactor = 1.0;
        % target size at the initial scale
        base_target_sz = target_sz / currentScaleFactor;
        
        % window size, taking padding into account
        switch tracker.search_area_shape
            case 'proportional'
                sz = floor( base_target_sz * search_area_scale);     % proportional area, same aspect ratio as the target
            case 'square'
                sz = repmat(sqrt(prod(base_target_sz * search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
            case 'fix_padding'
                sz = base_target_sz + sqrt(prod(base_target_sz * search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
            otherwise
                error('Unknown "tracker.search_area_shape". Must be ''proportional'', ''square'' or ''fix_padding''');
        end

        % set the size to exactly match the cell size
        sz = round(sz / featureRatio) * featureRatio;
        use_sz = floor(sz/featureRatio);
        
        % construct the label function- correlation output, 2D gaussian function,
        % with a peak located upon the target
        output_sigma = sqrt(prod(floor(base_target_sz/featureRatio))) * output_sigma_factor;
        rg           = circshift(-floor((use_sz(1)-1)/2):ceil((use_sz(1)-1)/2), [0 -floor((use_sz(1)-1)/2)]);
        cg           = circshift(-floor((use_sz(2)-1)/2):ceil((use_sz(2)-1)/2), [0 -floor((use_sz(2)-1)/2)]);
        [rs, cs]     = ndgrid( rg,cg);
        y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
        yf           = fft2(y); %   FFT of y.

        if interpolate_response == 1
            tracker.interp_sz = use_sz * featureRatio;
        else
            tracker.interp_sz = use_sz;
        end

        % construct cosine window
        cos_window = single(hann(use_sz(1))*hann(use_sz(2))');
        
        % Calculate feature dimension
        colorImage = true;
        % compute feature dimensionality
        feature_dim = 0;
        for n = 1:length(features)

            if ~isfield(features{n}.fparams,'useForColor')
                features{n}.fparams.useForColor = true;
            end

            if ~isfield(features{n}.fparams,'useForGray')
                features{n}.fparams.useForGray = true;
            end

            if (features{n}.fparams.useForColor && colorImage) || (features{n}.fparams.useForGray && ~colorImage)
                feature_dim = feature_dim + features{n}.fparams.nDim;
            end
        end
        
if nScales > 0  
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
    scaleFactors = scale_step .^ scale_exp;
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(I,1) size(I,2)] ./ base_target_sz)) / log(scale_step));
end

        if interpolate_response >= 3
            % Pre-computes the grid that is used for socre optimization
            ky = circshift(-floor((use_sz(1) - 1)/2) : ceil((use_sz(1) - 1)/2), [1, -floor((use_sz(1) - 1)/2)]);
            kx = circshift(-floor((use_sz(2) - 1)/2) : ceil((use_sz(2) - 1)/2), [1, -floor((use_sz(2) - 1)/2)])';
            newton_iterations = opt.newton_iterations;
            tracker.ky = ky;
            tracker.kx = kx;
        end
        
        % allocate memory for multi-scale tracking
        multires_pixel_template = zeros(sz(1), sz(2), size(I,3), nScales, 'uint8');
        small_filter_sz = floor(base_target_sz/featureRatio);
   
        %% extract some feature to form model
        % extract training sample image region
        pixels = get_pixels(I,pos,round(sz*currentScaleFactor),sz);
        
        if(size(pixels,1)~=sz(1)||size(pixels,2)~=sz(2))    %me kcf add
            pixels = imResample(pixels, sz, 'bilinear');
        end

        % extract features and do windowing
        xf = fft2(bsxfun(@times,get_features(pixels,features,global_feat_params),cos_window));
        
        model_xf = xf;
        
        g_f = single(zeros(size(xf)));
        h_f = g_f;
        l_f = g_f;
        mu    = 1;
        betha = 10;
        mumax = 10000;
        i = 1;
    
        T = prod(use_sz);
        S_xx = sum(conj(model_xf) .* model_xf, 3);
        tracker.admm_iterations = 2;
        %   ADMM
        while (i <= tracker.admm_iterations)
            %   solve for G- please refer to the paper for more details
            B = S_xx + (T * mu);
            S_lx = sum(conj(model_xf) .* l_f, 3);
            S_hx = sum(conj(model_xf) .* h_f, 3);
            g_f = (((1/(T*mu)) * bsxfun(@times, yf, model_xf)) - ((1/mu) * l_f) + h_f) - ...
                bsxfun(@rdivide,(((1/(T*mu)) * bsxfun(@times, model_xf, (S_xx .* yf))) - ((1/mu) * bsxfun(@times, model_xf, S_lx)) + (bsxfun(@times, model_xf, S_hx))), B);

            %   solve for H
            h = (T/((mu*T)+ tracker.admm_lambda))* ifft2((mu*g_f) + l_f);
            [sx,sy,h] = get_subwindow_no_window(h, floor(use_sz/2) , small_filter_sz);
            t = single(zeros(use_sz(1), use_sz(2), size(h,3)));
            t(sx,sy,:) = h;
            h_f = fft2(t);

            %   update L
            l_f = l_f + (mu * (g_f - h_f));

            %   update mu- betha = 10.
            mu = min(betha * mu, mumax);
            i = i+1;
        end
        
        target_sz = floor(base_target_sz * currentScaleFactor);
        
        tracker.init_target_sz = target_sz;
        tracker.g_f = g_f;      % model
        tracker.model_xf = model_xf;
        tracker.cos_window = cos_window;
        tracker.use_sz = use_sz;
        tracker.yf = yf;
        tracker.y = y;
        tracker.sz = sz;
        tracker.small_filter_sz= small_filter_sz;
        tracker.scale = scale;
        tracker.scaleFactors = scaleFactors;
        
        tracker.min_scale_factor = min_scale_factor;
        tracker.max_scale_factor = max_scale_factor;
%         tracker.currentScaleFactor= currentScaleFactor;

    case 'CFNet'
        stats = opt.stats;
        net_base = '/home/qi/projects/MDP_KCF_Qi/CFNet/pretrained/networks/'; 
        % Load pre-trained network
        if ischar(tracker.net)
            % network has been passed as string
            net_path = [net_base tracker.net];
            net_z = load(net_path,'net');
        else
            % network has been passed as object
            net_z = tracker.net;
        end

        target_sz = [bb(4)-bb(2),bb(3)-bb(1)];
        pos = [bb(2), bb(1)] + target_sz/2;
        
        net_x = net_z;
        % Load a second copy of the network for the second branch
        net_z = dagnn.DagNN.loadobj(net_z.net);
        
        % Sanity check
        switch tracker.join.method
            case 'xcorr'
                assert(~find_layers_from_prefix(net_z, 'join_cf'), 'Check your join.method');
            case 'corrfilt'
                assert(find_layers_from_prefix(net_z, 'join_cf'), 'Check your join.method');
        end
        
        % create a full copy of the network, not just of the handle
        net_x = dagnn.DagNN.loadobj(net_x.net);
        
        net_z = init_net(net_z, tracker.gpus, tracker.init_gpu);
        net_x = init_net(net_x, tracker.gpus, false);
        
        % visualize net before trimming
        % display_net(net_z, {'exemplar', [255 255 3 8], 'instance', [255 255 3 8]}, 'net_full')
        %% Divide the net in 2
        % exemplar branch (only once per video) computes features for the target
        for i=1:numel(tracker.trim_x_branch)
            remove_layers_from_prefix(net_z, tracker.trim_x_branch{i});
        end
        % display_net(net_z, {'exemplar', [255 255 3 8]}, 'z_net')
        % display_net(net_z, {'exemplar', [127 127 3 8], 'target', [6 6 1 8]}, 'z_net')
        % instance branch computes features for search region and cross-correlates with z features
        for i=1:numel(tracker.trim_z_branch)
            remove_layers_from_prefix(net_x, tracker.trim_z_branch{i});
        end
        % display_net(net_x, {'instance', [255 255 3 8], 'br1_out', [30 30 32 8]}, 'x_net')
        % display_net(net_x, {'instance', [255 255 3 8], 'join_tmpl_cropped', [17 17 32 8]}, 'x_net')

        z_out_id = net_z.getOutputs();
        %%
        if ~isempty(tracker.gpus)
            I = gpuArray(I);
        end
        % if grayscale repeat one channel to match filters size
        if(size(I, 3)==1)
            I = repmat(I, [1 1 3]);
        end
        
        avgChans = gather([mean(mean(I(:,:,1))) mean(mean(I(:,:,2))) mean(mean(I(:,:,3)))]);
        
        wc_z = target_sz(2) + tracker.contextAmount*sum(target_sz);
        hc_z = target_sz(1) + tracker.contextAmount*sum(target_sz);
        s_z = sqrt(wc_z*hc_z);
        s_x = tracker.instanceSize/tracker.exemplarSize * s_z;
        scales = (tracker.scaleStep .^ ((ceil(tracker.numScale/2)-tracker.numScale) : floor(tracker.numScale/2)));
        scaledExemplar = s_z .* scales;
        
        % initialize the exemplar
        [z_crop, ~] = make_scale_pyramid(I, pos, scaledExemplar, tracker.exemplarSize, avgChans, stats, tracker);
        z_crop = z_crop(:,:,:,ceil(tracker.numScale/2));

        if tracker.subMean
            z_crop = bsxfun(@minus, z_crop, reshape(stats.z.rgbMean, [1 1 3]));
        end
        
        net_z.eval({'exemplar', z_crop});
        get_vars = @(net, ids) cellfun(@(id) net.getVar(id).value, ids, 'UniformOutput', false);
        z_out_val = get_vars(net_z, z_out_id);
            
        min_s_x = tracker.minSFactor*s_x;
        max_s_x = tracker.maxSFactor*s_x;
        min_s_z = tracker.minSFactor*s_z;
        max_s_z = tracker.maxSFactor*s_z;
        
        switch tracker.join.method
        case 'corrfilt'
            tracker.id_score = 'join_out';
            % Extract scores for join_out (pre fin_adjust to have them in the range 0-1)
            net_x.vars(end-1).precious = true;
        end
    
        % windowing to penalize large displacements
        window = single(hann(tracker.scoreSize*tracker.responseUp) * hann(tracker.scoreSize*tracker.responseUp)');
        window = window / sum(window(:));
        
        scoreId = net_x.getVarIndex(tracker.id_score);
        
        % Then we go to Tracker main loop
        tracker.scoreId = scoreId;
        tracker.window = window;
        tracker.s_x = s_x;
        tracker.scales = scales;
        tracker.z_out_val = z_out_val;
        tracker.net_x = net_x;
        tracker.s_z = s_z;
        tracker.min_s_x = min_s_x;
        tracker.max_s_x = max_s_x;
        tracker.min_s_z = min_s_z;
        tracker.max_s_z = max_s_z;
        tracker.z_out_id = z_out_id;
        tracker.net_z = net_z;
        tracker.avgChans = avgChans;
        tracker.get_vars = get_vars;        
        
    case 'lomo'
        
    case 'SCT'
        % Feature & Kernel parameters setting
        interp_factor = tracker.interp_factor;
        kernel.sigma = tracker.kernel.sigma;

        kernel.poly_a = tracker.kernel.poly_a;
        kernel.poly_b = tracker.kernel.poly_b;

        features.gray = tracker.features.gray;
        features.hog = tracker.features.hog;
        features.hog_orientations = tracker.features.hog_orientations;
        cell_size = tracker.cell_size;

        % KCF parameters setting
        padding = tracker.padding  %extra area surrounding the target
        lambda = tracker.lambda;  %regularization
        output_sigma_factor = tracker.output_sigma_factor;  %spatial bandwidth (proportional to target)

        % Attention map parameters setting
        Nfo = tracker.Nfo;
        boundary_ratio = tracker.boundary_ratio;
        salWeight = tracker.salWeight;
        bSal = tracker.bSal;

        % multiple module trackers type initialization
        filterPool(1).kernelType = tracker.filterPool(1).kernelType;
        filterPool(1).featureType = tracker.filterPool(1).featureType;
        filterPool(2).kernelType = tracker.filterPool(2).kernelType;
        filterPool(2).featureType = tracker.filterPool(2).featureType;
        filterPool(3).kernelType = tracker.filterPool(3).kernelType;
        filterPool(3).featureType = tracker.filterPool(3).featureType;
        filterPool(4).kernelType = tracker.filterPool(4).kernelType;
        filterPool(4).featureType = tracker.filterPool(4).featureType;
        
        %set initial position and size
        target_sz = [bb(4)-bb(2),bb(3)-bb(1)];
        pos = [bb(2), bb(1)] + target_sz/2;
        
        %% Tracker initialization
        %if the target is large, lower the resolution
        resize_image = (sqrt(prod(target_sz)) >= 100);
        if resize_image,
            pos = floor(pos / 2);
            target_sz = floor(target_sz / 2);
        end

        window_sz = floor(target_sz * (1 + padding));

        % Initialize the constant values & maps
        output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
        yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));
        cos_window = (hann(size(yf,1)) * hann(size(yf,2))');
        mask = ones(size(yf,1), size(yf,2)); % Initial mask for strong saliency map
        depthBoundaryX = max(round(size(yf,2)*boundary_ratio), 3);
        depthBoundaryY = max(round(size(yf,1)*boundary_ratio), 3);
        mask( depthBoundaryY:(end-depthBoundaryY+1), depthBoundaryX:(end-depthBoundaryX+1) ) = 0;
        
        %load image
        im = I; % gray image for HOG feature
        im2 = im; % color or gray image...Color im2
        if size(im,3) > 1,
            im = rgb2gray(im);
        end
        if resize_image,
            im = imresize(im, 0.5);
            im2 = imresize(im2, 0.5);
        end
        
        % HOG feature extraction
        patch = get_subwindow_SCT(im, pos, window_sz);
        x = get_features_SCT(patch, features, cell_size, []);
        % Color/gray intensity feature extraction ..Color img2
        patch2 = get_subwindow_SCT(im2, pos, window_sz);
        feature = double(imresize(patch2, [size(x,1), size(x,2)]))/255;
        if(size(feature,3) > 1)
            feature =  cat(3, feature, rgb2lab(feature) / 255 + 0.5);
        end
        x2 = feature;
        
        [rf{1,1}, stS{1,1}] = init_stSaliency(x2, mask);
        saliencyMap{1,1} = cos_window;
        [rf{1,2}, stS{1,2}] = init_stSaliency(x, mask);
        saliencyMap{1,2} = cos_window;
        
        stS{1,1} = saliencyMap{1,1};
        stS{1,2} = saliencyMap{1,2};
        
        % attention map multiplication for tracking
        x = bsxfun(@times, stS{1,2}, x);
        x2 = bsxfun(@times, stS{1,1}, x2);
        xf = fft2(x);
        xf2 = fft2(x2);
        
        % Module-wise training
        for ii = 1:4
            switch filterPool(ii).kernelType
                case 'gaussian',
                    if strcmp(filterPool(ii).featureType, 'hog')
                        filterPool(ii).kf = gaussian_correlation(xf, xf, kernel.sigma);
                    else
                        filterPool(ii).kf = gaussian_correlation(xf2, xf2, kernel.sigma);
                    end
                case 'polynomial',
                    if strcmp(filterPool(ii).featureType, 'hog')
                        filterPool(ii).kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
                    else
                        filterPool(ii).kf = polynomial_correlation(xf2, xf2, kernel.poly_a, kernel.poly_b);
                    end
                case 'linear',
                    if strcmp(filterPool(ii).featureType, 'hog')
                        filterPool(ii).kf = linear_correlation(xf, xf);
                    else
                        filterPool(ii).kf = linear_correlation(xf2, xf2);
                    end
            end
            filterPool(ii).dalphaf = 1 ./ (filterPool(ii).kf + lambda);   %equation for fast training
        end
        
        % Temporal association (filter update)
        for ii = 1:4
            filterPool(ii).model_dalphaf = filterPool(ii).dalphaf;
            if strcmp(filterPool(ii).featureType, 'hog')
                filterPool(ii).model_xf = xf;
            else
                filterPool(ii).model_xf = xf2;
            end
        end
        
        % Estimate the priority & reliability for each module
        errs = zeros(4,4);
        errWeight = ones(size(yf));
        bin = ones(1,4);

        errMaps = zeros(size(yf,1), size(yf,2) ,4);
        for ii = 1:4
            errMaps(:,:,ii) = (real(ifft2(filterPool(ii).kf .* yf .* filterPool(ii).model_dalphaf - yf))).^2;
        end
        
        for jj = 1:4

            if(jj < 4)
                % estimate the module-wise error
                for ii = 1:4
                    errs(jj,ii) = sqrt(sum( vec(errWeight.*errMaps(:,:,ii)) ));
                end

                % Find the next best module
                idx = find(errs(jj,:) == min(errs(jj,bin==1)));
                idx = idx(1);

                bin(idx) = 0;

                % For reliability, the error weight is estimated
                errWeight = errWeight .* exp((errMaps(:,:,idx) / max(vec(errMaps(:,:,idx)))));
                errWeight = errWeight / max(vec(errWeight));

            else
                idx = find(bin==1);
            end

            % For priority, the order of the modules changes
            multiFilters(jj).kernelType = filterPool(idx).kernelType;
            multiFilters(jj).featureType = filterPool(idx).featureType;
            multiFilters(jj).model_alphaf = yf .* filterPool(idx).model_dalphaf;
            multiFilters(jj).model_xf = filterPool(idx).model_xf;
            multiFilters(jj).weight = exp(-0.01*errs(1,idx));

        end
        
        % bulabula
        tracker.resize_image = resize_image;
        tracker.window_sz = window_sz;
        tracker.features = features;
        tracker.cell_size = cell_size;
        tracker.bSal = bSal;        
        tracker.salWeight = salWeight;
        tracker.cos_window = cos_window;
        tracker.yf = yf;
        tracker.multiFilters = multiFilters;        
        tracker.kernel = kernel;
        tracker.mask = mask;
        tracker.rf = rf;
        tracker.stS = stS;
        tracker.saliencyMap = saliencyMap;
        tracker.filterPool = filterPool;
        tracker. kernel =  kernel;
        
    case 'ADNet'
        
%         % init containers
%         bboxes = zeros(size(vid_info.gt));
%         total_pos_data = cell(1,1,1,vid_info.nframes);
%         total_neg_data = cell(1,1,1,vid_info.nframes);
%         total_pos_action_labels = cell(1,vid_info.nframes);
%         total_pos_examples = cell(1,vid_info.nframes);
%         total_neg_examples = cell(1,vid_info.nframes);         
        
        
        % net
        load('ADNet/models/net_rl.mat');
        
        % init model networks
        net.layers(net.getLayerIndex('loss')).block.loss = 'softmaxlog';
        [net, net_conv, net_fc] = split_dagNN(net);

        net_fc.params(net_fc.getParamIndex('fc6_1b')).value = ...
            gpuArray(ones(size(net_fc.params(net_fc.getParamIndex('fc6_1b')).value), 'single') * 0.01);

        for p = 1 : numel(net_fc.params)
            if mod(p, 2) == 1
                net_fc.params(p).learningRate = 10;
            else
                net_fc.params(p).learningRate = 20;
            end
        end
        net_fc.params(net_fc.getParamIndex('fc6_1f')).learningRate = 20;
        net_fc.params(net_fc.getParamIndex('fc6_1b')).learningRate = 40;


        obj_idx = net_fc.getVarIndex('objective');
        obj_score_idx = net_fc.getVarIndex('objective_score');
        pred_idx = net_fc.getVarIndex('prediction');
        pred_score_idx = net_fc.getVarIndex('prediction_score');
        conv_feat_idx = net_conv.getVarIndex('x10');
        net_fc.vars(obj_idx).precious = 1;
        net_fc.vars(obj_score_idx).precious = 1;
        net_fc.vars(pred_idx).precious = 1;
        net_fc.vars(pred_score_idx).precious = 1;
        net_conv.vars(conv_feat_idx).precious = 1;

        net.move('gpu');
        net_conv.move('gpu');
        net_fc.move('gpu');
        
        % RUN TRACKING
        cont_negatives = 0;
        frame_window = [];
        
        % ============================
        % LOAD & FINETUNE FC NETWORKS
        % ============================
        target_sz = [bb(4)-bb(2),bb(3)-bb(1)];
        pos = [bb(2), bb(1)] + target_sz/2;
        
        curr_bbox = bb;
        curr_img = I;
        
        if(size(curr_img,3)==1), curr_img = cat(3,curr_img,curr_img,curr_img); end
        
        imSize = size(curr_img);
        
        opt.imgSize = imSize;
        tracker.imgSize = opt.imgSize;
        
        action_history_oh_zeros = zeros(opt.num_actions*opt.num_action_history, 1);
        action_history_oh = action_history_oh_zeros;
        
        % generate samples
        pos_examples = single(gen_samples('gaussian', curr_bbox, opt.nPos_init*2, opt, opt.finetune_trans, opt.finetune_scale_factor));
        r = overlap_ratio(pos_examples,curr_bbox);
        pos_examples = pos_examples(r>opt.posThre_init,:);
        pos_examples = pos_examples(randsample(end,min(opt.nPos_init,end)),:);
        neg_examples = [gen_samples('uniform', curr_bbox, opt.nNeg_init, opt, 1, 10);...
            gen_samples('whole', curr_bbox, opt.nNeg_init, opt)];
        r = overlap_ratio(neg_examples,curr_bbox);
        neg_examples = single(neg_examples(r<opt.negThre_init,:));
        neg_examples = neg_examples(randsample(end,min(opt.nNeg_init,end)),:);
        examples = [pos_examples; neg_examples];
        pos_idx = 1:size(pos_examples,1);
        neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);
        net_conv.mode = 'test';
        feat_conv = get_conv_feature(net_conv, curr_img, examples, opt);
        pos_data = feat_conv(:,:,:,pos_idx);
        neg_data = feat_conv(:,:,:,neg_idx);
        % get action labels
        pos_action_labels = gen_action_labels(opt.num_actions, opt, pos_examples, curr_bbox);
        
        opt.maxiter = opt.finetune_iters;

        
        opt.learningRate = 0.0003;

        
        net_fc.mode = 'test';
        
        net_fc_noah = copy(net_fc);
        net_fc_noah.removeLayer('concat');
        net_fc_noah.layers(4).inputs = 'x13';
        net_fc_noah.params(3).value(:,:,512+1:end,:) = [];
        net_fc_noah.rebuild();
        net_fc_noah.move('gpu');
        net_fc.move('gpu');
        
        [net_fc_noah, ~] =  train_fc_finetune_hem(net_fc_noah, opt, ...
            pos_data, neg_data, pos_action_labels);
        for fci = 1 : 8
            if fci == 3
                net_fc.params(fci).value(:,:,1:512,:) = net_fc_noah.params(fci).value;
            else
                net_fc.params(fci).value = net_fc_noah.params(fci).value;
            end
        end
        
        %1==>frame_id
        total_pos_data{frame_id} = pos_data;
        total_neg_data{frame_id} = neg_data;
        total_pos_action_labels{frame_id} = pos_action_labels;
        total_pos_examples{frame_id} = pos_examples';
        total_neg_examples{frame_id} = neg_examples';
        
        frame_window = [frame_window, frame_id];
        is_negative = false;       
        
        action_history = zeros(opt.num_show_actions, 1);
        this_actions = zeros(opt.num_show_actions, 1);
        
%         tracker.net = net;
        tracker.net_conv = net_conv;
        tracker.net_fc = net_fc;
        
        tracker.total_pos_data = total_pos_data;
        tracker.total_neg_data = total_neg_data;
        tracker.total_pos_action_labels = total_pos_action_labels;
        tracker.total_pos_examples = total_pos_examples;
        tracker.total_neg_examples = total_neg_examples;
        tracker.frame_window = frame_window;
        tracker.action_history = action_history;
        tracker.action_history_oh = action_history_oh;
        tracker.cont_negatives = cont_negatives;
        tracker.frame_window = frame_window;
        
    case 'lomocf'
        % KCF
        cell_size = tracker.cell_size;
        features = tracker.features;
        kernel = tracker.kernel;
        padding = tracker.padding;
        output_sigma_factor = tracker.output_sigma_factor;
        lambda = tracker.lambda;
        template_sz = tracker.template_sz;
        scale = 1;

        %set initial position and size
        target_sz = [bb(4)-bb(2),bb(3)-bb(1)];
        pos = [bb(2), bb(1)] + target_sz/2;

        %if the target is large, lower the resolution, we don't need that much
        %detail
        if (sqrt(prod(target_sz)) >= template_sz)
            if(target_sz(1)>target_sz(2))
                scale = target_sz(1)/template_sz;
            else
                scale = target_sz(2)/template_sz;
            end
        end
        target_sz = floor(target_sz/scale);

        %window size, taking padding into account
        window_sz = floor(target_sz * (1 + padding));

        %create regression labels, gaussian shaped, with a bandwidth
        %proportional to target size
        output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
        yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));

        %store pre-computed cosine window
        cos_window = hann(size(yf,1)) * hann(size(yf,2))';	

        if size(I,3) > 1,
            I = rgb2gray(I);
        end

        extracted_sz = floor(window_sz * scale);

        %obtain a subwindow for training at newly estimated target position
        patch = get_subwindow_kcf(I, pos, extracted_sz);
        if(size(patch,1)~=window_sz(1)||size(patch,2)~=window_sz(2))
            patch = imResample(patch, window_sz, 'bilinear');
        end
        xf = fft2(get_features_kcf(patch, features, cell_size, cos_window));

        kf = gaussian_correlation(xf, xf, kernel.sigma);
        alphaf = yf ./ (kf + lambda);   %equation for fast training
        model_alphaf = alphaf;
        model_xf = xf;

        tracker.init_target_sz = target_sz;
        tracker.model_alphaf = model_alphaf;
        tracker.model_xf = model_xf;
        tracker.cos_window = cos_window;
        tracker.window_sz = window_sz;
        tracker.yf = yf;
        tracker.scale = scale;
        
end % //switch end
end % //FUNCTION END

