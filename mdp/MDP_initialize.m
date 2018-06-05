% --------------------------------------------------------
% MDP Tracking
% Copyright (c) 2015 CVGL Stanford
% Licensed under The MIT License [see LICENSE for details]
% Written by Yu Xiang
% --------------------------------------------------------
%
% initialization of the tracker
function tracker = MDP_initialize(I, dres_det, labels, opt)

image_width = size(I,2);
image_height = size(I,1);

% normalization factor for features
tracker.image_width = image_width;
tracker.image_height = image_height;
tracker.max_width = max(dres_det.w);
tracker.max_height = max(dres_det.h);
tracker.max_score = max(dres_det.r);
tracker.fb_factor = opt.fb_factor;

% active
tracker.fnum_active = 6;
factive = MDP_feature_active(tracker, dres_det);
index = labels ~= 0;
tracker.factive = factive(index,:);
tracker.lactive = labels(index);
tracker.w_active = svmtrain(tracker.lactive, tracker.factive, '-c 1 -q');

% initial state
tracker.prev_state = 1;
tracker.state = 1;

% association model
tracker.fnum_tracked = 2;

switch opt.method
    case 'lomocf'        
        tracker.fnum_occluded = 14;        %---------------------lomocf
%         tracker.fnum_occluded = 13;
    case 'ori'
        tracker.fnum_occluded = 12;
    otherwise
        tracker.fnum_occluded = 13;        %---------------------cf
        % tracker.fnum_occluded = 12;
end

tracker.w_occluded = [];
tracker.f_occluded = [];
tracker.l_occluded = [];
tracker.streak_occluded = 0;

% tracker parameters
tracker.num = opt.num;
tracker.threshold_ratio = opt.threshold_ratio;
tracker.threshold_dis = opt.threshold_dis;
tracker.threshold_box = opt.threshold_box;
tracker.std_box = opt.std_box;  % [width height]
tracker.margin_box = opt.margin_box;
tracker.enlarge_box = opt.enlarge_box;
tracker.level_track = opt.level_track;
tracker.level = opt.level;
tracker.max_ratio = opt.max_ratio;
tracker.min_vnorm = opt.min_vnorm;
tracker.overlap_box = opt.overlap_box;
tracker.patchsize = opt.patchsize;
tracker.weight_tracking = opt.weight_tracking;
tracker.weight_detection = opt.weight_detection;   % add
tracker.weight_association = opt.weight_association;

%% choose an appearence CF                      %---------------------------------CF
switch opt.method
    case 'KCF'
        %% kcf parameters
        tracker.cell_size = opt.cell_size;
        tracker.features = opt.features;
        tracker.kernel = opt.kernel;
        tracker.padding = opt.padding;
        tracker.output_sigma_factor = opt.output_sigma_factor;
        tracker.interp_factor_1=opt.interp_factor_1;
        tracker.interp_factor_2=opt.interp_factor_2;
        tracker.lambda = opt.lambda;
        tracker.template_sz = opt.template_sz;
    case 'BACF'
        %   Global feature parameters 
        tracker.features = opt.t_features;
        tracker.t_global.cell_size = opt.t_global.cell_size;                  % Feature cell size
        tracker.t_global.cell_selection_thresh = opt.t_global.cell_selection_thresh; % Threshold for reducing the cell size in low-resolution cases

        %   Search region + extended background parameters
        tracker.search_area_shape = opt.search_area_shape;    % the shape of the training/detection window: 'proportional', 'square' or 'fix_padding'
        tracker.search_area_scale = opt.search_area_scale;           % the size of the training/detection area proportional to the target size
        tracker.filter_max_area = opt.filter_max_area;        % the size of the training/detection area in feature grid cells

        %   Learning parameters
        tracker.learning_rate = opt.learning_rate;        % learning rate
        tracker.learning_rate2 = opt.learning_rate2;
        tracker.output_sigma_factor = opt.output_sigma_factor;		% standard deviation of the desired correlation output (proportional to target)

        %   Detection parameters
        tracker.interpolate_response = opt.interpolate_response;        % correlation score interpolation strategy: 0 - off, 1 - feature grid, 2 - pixel grid, 4 - Newton's method
        tracker.newton_iterations = opt.newton_iterations;           % number of Newton's iteration to maximize the detection scores
                        % the weight of the standard (uniform) regularization, only used when opt.use_reg_window == 0
        %   Scale parameters
        tracker.number_of_scales = opt.number_of_scales;
        tracker.scale_step = opt.scale_step;

        %   ADMM parameters, # of iteration, and lambda- mu and betha are set in
        %   the main function.
        tracker.admm_iterations = opt.admm_iterations;
        tracker.admm_lambda = opt.admm_lambda;

        tracker.template_sz = opt.template_sz;
    case 'CFNet'
        %% default hyper-params for SiamFC tracker.
        tracker.join.method = 'corrfilt';
        tracker.net = 'cfnet-conv2_e80.mat';
        tracker.net_gray = 'cfnet-conv2_gray_e40.mat';
        tracker.numScale = 3;
        tracker.scaleStep = 1.0575;
        tracker.scalePenalty = 0.9780;
        tracker.scaleLR = 0.52;
        tracker.responseUp = 8;
        tracker.wInfluence = 0.2625; % influence of cosine window for displacement penalty
        tracker.minSFactor = 0.2;
        tracker.maxSFactor = 5;
        tracker.zLR = 0.005; % update rate of the exemplar for the rolling avg (use very low values <0.015)
        tracker.video = '';
        tracker.visualization = false;
        tracker.gpus = [];                        % default 1  not[]
        tracker.track_lost = [];
        tracker.startFrame = 1;
        tracker.fout = -1;
        tracker.imgFiles = [];
        tracker.targetPosition = [];
        tracker.targetSize = [];
        tracker.track_lost = [];
        tracker.ground_truth = [];
        
        %% params from the network architecture params (TODO: should be inferred from the saved network)
    % they have to be consistent with the training
        tracker.scoreSize = 33;
        tracker.totalStride = 4;
        tracker.contextAmount = 0.5; % context amount for the exemplar
        tracker.subMean = false;
        % prefix and ids
        tracker.prefix_z = 'br1_'; % used to identify the layers of the exemplar
        tracker.prefix_x = 'br2_'; % used to identify the layers of the instance
        tracker.id_score = 'score';
        tracker.trim_z_branch = {'br1_'};
        tracker.trim_x_branch = {'br2_','join_xcorr','fin_adjust'};
        tracker.init_gpu = true;
        % Get environment-specific default paths.
%         tracker.paths = struct();   % I do not use it.
%         tracker.paths = env_paths_tracking(tracker.paths);
%         tracker = vl_argparse(tracker, varargin);
        
        % network surgeries depend on the architecture    
        switch tracker.join.method
            case 'xcorr'
                tracker.trim_x_branch = {'br2_','join_xcorr','fin_'};
                tracker.trim_z_branch = {'br1_'};
                tracker.exemplarSize = 127;
                tracker.instanceSize = 255;
            case 'corrfilt'
                tracker.trim_x_branch = {'br2_','join_xcorr','fin_adjust'};
                tracker.trim_z_branch = {'br1_','join_cf','join_crop_z'};
                tracker.exemplarSize = 255;
                tracker.instanceSize = 255;
            otherwise
                error('network type unspecified');
        end
        
        % Load ImageNet Video statistics
        statspath ='/home/qi/projects/MDP_LOMOCF_Qi/CFNet/data/ILSVRC2015.stats.mat';
        opt.stats = load(statspath);
        tracker.stats = load(statspath);
        
    case 'lomo'
        tracker.numScales = opt.numScales;     %..
        tracker.blockSize = opt.blockSize;     %..
        tracker.blockStep = opt.blockStep;
        tracker.hsvBins = opt.hsvBins;
        tracker.tau = opt.tau;
        tracker.R = opt.R;                     %..
        tracker.numPoints = opt.numPoints;
    
    case 'SCT'
        % Feature & Kernel parameters setting
        tracker.interp_factor = opt.interp_factor;
        tracker.kernel.sigma = opt.kernel.sigma;

        tracker.kernel.poly_a = opt.kernel.poly_a;
        tracker.kernel.poly_b = opt.kernel.poly_b;

        tracker.features.gray = opt.features.gray;
        tracker.features.hog = opt.features.hog;
        tracker.features.hog_orientations = opt.features.hog_orientations;
        tracker.cell_size = opt.cell_size;

        % KCF parameters setting
        tracker.padding = opt.padding;  %extra area surrounding the target
        tracker.lambda = opt.lambda;  %regularization
        tracker.output_sigma_factor = opt.output_sigma_factor;  %spatial bandwidth (proportional to target)

        % Attention map parameters setting
        tracker.Nfo = opt.Nfo;
        tracker.boundary_ratio = opt.boundary_ratio;
        tracker.salWeight = opt.salWeight;
        tracker.bSal = opt.bSal;

        % multiple module trackers type initialization
        tracker.filterPool(1).kernelType = opt.filterPool(1).kernelType;
        tracker.filterPool(1).featureType = opt.filterPool(1).featureType;
        tracker.filterPool(2).kernelType = opt.filterPool(2).kernelType;
        tracker.filterPool(2).featureType = opt.filterPool(2).featureType;
        tracker.filterPool(3).kernelType = opt.filterPool(3).kernelType;
        tracker.filterPool(3).featureType = opt.filterPool(3).featureType;
        tracker.filterPool(4).kernelType = opt.filterPool(4).kernelType;
        tracker.filterPool(4).featureType = opt.filterPool(4).featureType;
    
    case ' ADNet'
        % parameter settings
        % show_visualization = 0;
        % record_video = 0;
        % GT_anno_interval = 1;

        % ============================
        % NETWORK PARAMETERS
        % ============================
        tracker.train.weightDecay = opt.train.weightDecay;
        tracker.train.momentum = opt.train.momentum ;
        tracker.train.learningRate = opt.train.learningRate;
        tracker.train.conserveMemory = opt.train.conserveMemory ;
        tracker.minibatch_size = opt.minibatch_size;
        tracker.numEpoch = opt.numEpoch;
        tracker.numInnerEpoch = opt.numInnerEpoch;
        tracker.continueTrain = opt.continueTrain;
        tracker.samplePerFrame_large = opt.samplePerFrame_large;
        tracker.samplePerFrame_small = opt.samplePerFrame_small;
        tracker.inputSize = opt.inputSize;
        tracker.stopIou = opt.stopIou;
        tracker.meta.inputSize = opt.meta.inputSize;

        tracker.train.gt_skip = opt.train.gt_skip;
        tracker.train.rl_num_batches = opt.train.rl_num_batches;

        tracker.train.RL_steps = opt.train.RL_steps;


        tracker.use_finetune = opt.use_finetune;

        tracker.scale_factor = opt.scale_factor;

        % test
        tracker.finetune_iters = opt.finetune_iters;
        tracker.finetune_iters_online = opt.finetune_iters_online;
        tracker.finetune_interval = opt.finetune_interval;
        tracker.posThre_init = opt.posThre_init;
        tracker.negThre_init = opt.negThre_init;
        tracker.posThre_online = opt.posThre_online;
        tracker.negThre_online = opt.negThre_online;
        tracker.nPos_init = opt.nPos_init;
        tracker.nNeg_init = opt.nNeg_init;
        tracker.nPos_online = opt.nPos_online;
        tracker.nNeg_online = opt.nNeg_online;
        tracker.finetune_scale_factor = opt.finetune_scale_factor;
        tracker.redet_scale_factor = opt.redet_scale_factor;
        tracker.finetune_trans = opt.finetune_trans;
        tracker.redet_samples = opt.redet_samples;



        tracker.successThre = opt.successThre;
        tracker.failedThre = opt.failedThre;

        tracker.nFrames_long = opt.nFrames_long; % long-term period
        tracker.nFrames_short = opt.nFrames_short; % short-term period

        tracker.nPos_train = opt.nPos_train;
        tracker.nNeg_train = opt.nNeg_train;
        tracker.posThre_train = opt.posThre_train;
        tracker.negThre_train = opt.negThre_train;

        tracker.random_perturb.x = opt.random_perturb.x;
        tracker.random_perturb.y = opt.random_perturb.y;
        tracker.random_perturb.w = opt.random_perturb.w;
        tracker.random_perturb.h = opt.random_perturb.h;
        tracker.action_move.x = opt.action_move.x;
        tracker.action_move.y = opt.action_move.y;
        tracker.action_move.w = opt.action_move.w;
        tracker.action_move.h = opt.action_move.h;

        tracker.action_move.deltas = opt.action_move.deltas;

        tracker.num_actions = opt.num_actions;
        tracker.stop_action = opt.stop_action;
        tracker.num_show_actions = opt.num_show_actions;
        tracker.num_action_step_max = opt.num_action_step_max;
        tracker.num_action_history = opt.num_action_history;

        tracker.visualize = opt.visualize;
        tracker.printscreen = opt.printscreen;
        
    case 'lomocf'
        
        % kcf parameters
        tracker.cell_size = opt.cell_size;
        tracker.features = opt.features;
        tracker.kernel = opt.kernel;
        tracker.padding = opt.padding;
        tracker.output_sigma_factor = opt.output_sigma_factor;
        tracker.interp_factor_1=opt.interp_factor_1;
        tracker.interp_factor_2=opt.interp_factor_2;
        tracker.lambda = opt.lambda;
        tracker.template_sz = opt.template_sz;
        
        %lomo parameters
        tracker.numScales = opt.numScales;     %..
        tracker.blockSize = opt.blockSize;     %..
        tracker.blockStep = opt.blockStep;
        tracker.hsvBins = opt.hsvBins;
        tracker.tau = opt.tau;
        tracker.R = opt.R;                     %..
        tracker.numPoints = opt.numPoints;
        
    case 'ori'
        
end


% display results or not
tracker.is_show = opt.is_show;