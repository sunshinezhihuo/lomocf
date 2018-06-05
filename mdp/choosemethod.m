% choose an appearence method KCF BACF CFNet
% opt.method = 'KCF';
% opt.method = 'BACF';
% opt.method = 'CFNet';
% opt.method = 'lomo';
% opt.method = 'SCT';
% opt.method = 'ADNet';   % having some problems
opt.method = 'lomocf';
% opt.method = 'ori';

switch opt.method
    case 'KCF'
        %% parameters for kcf tracking   % ------add
        opt.max_iter = 30;     % max iterations in total        %------change
        opt.max_count = 5;       % max iterations per sequence  % -----change
        
        kernel.type = 'gaussian';
        kernel.sigma = 0.5;       %gaussian kernel bandwidth

        kernel.poly_a = 1;        %polynomial kernel additive term
        kernel.poly_b = 9;        %polynomial kernel exponent

        features.gray = false;
        features.hog = true;
        features.hog_orientations = 9;

        opt.cell_size = 4;
        opt.features = features;
        opt.kernel = kernel;
        opt.padding = 1.5;
        opt.output_sigma_factor = 0.1;
        opt.interp_factor_1 = 0.02;
        opt.interp_factor_2 = 0.1;
        opt.lambda = 1e-4;
        opt.template_sz = 96;
    case 'BACF'
%%
        %% parameters for BACF            % ------BACF
        opt.max_iter = 1500;     % max iterations in total        %------change
        opt.max_count = 20;       % max iterations per sequence  % -----change
        
        lr = 0.013;
        %   Default parameters used in the ICCV 2017 BACF paper

        %   HOG feature parameters
        hog_params.nDim   = 31;

        %   Grayscale feature parameters
        grayscale_params.colorspace='gray';
        grayscale_params.nDim = 1;

        %   Global feature parameters 
        opt.t_features = {
            ...struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...  % Grayscale is not used as default
            struct('getFeature',@get_fhog,'fparams',hog_params),...
        };
        opt.t_global.cell_size = 4;                  % Feature cell size
        opt.t_global.cell_selection_thresh = 0.75^2; % Threshold for reducing the cell size in low-resolution cases

        %   Search region + extended background parameters
        opt.search_area_shape = 'square';    % the shape of the training/detection window: 'proportional', 'square' or 'fix_padding'
        opt.search_area_scale = 5;           % the size of the training/detection area proportional to the target size
        opt.filter_max_area   = 50^2;        % the size of the training/detection area in feature grid cells

        %   Learning parameters
        opt.learning_rate       = lr;        % learning rate
        opt.learning_rate2      = 0.1;       % no use
        
        opt.output_sigma_factor = 1/16;		% standard deviation of the desired correlation output (proportional to target)

        %   Detection parameters
        opt.interpolate_response  = 0;        %4 correlation score interpolation strategy: 0 - off, 1 - feature grid, 2 - pixel grid, 4 - Newton's method
        opt.newton_iterations     = 50;           % number of Newton's iteration to maximize the detection scores
                        % the weight of the standard (uniform) regularization, only used when opt.use_reg_window == 0
        %   Scale parameters
        opt.number_of_scales =  1;     
        opt.scale_step       = 1.01;

        %   ADMM parameters, # of iteration, and lambda- mu and betha are set in
        %   the main function.
        opt.admm_iterations = 2;
        opt.admm_lambda = 0.01;


        opt.template_sz = 96;
    case 'CFNet'
               
        addpath(genpath('/home/qi/projects/MDP_LOMOCF_Qi/CFNet/matconvnet')); 
%         addpath(genpath('../util'));
        addpath(genpath('/home/qi/projects/MDP_LOMOCF_Qi/CFNet/src/tracking'));
        addpath(genpath('/home/qi/projects/MDP_LOMOCF_Qi/CFNet/src/util'));
        vl_setupnn;

        opt.max_iter = 200;
        opt.max_count = 7; 
        
        % Sample execution for CFNet-conv2
        % hyper-parameters reported in Supp.material for CVPR'17, Table 2 for arXiv version
        opt.join.method = 'corrfilt';
        opt.net = 'cfnet-conv2_e80.mat';
        opt.net_gray = 'cfnet-conv2_gray_e40.mat';
        opt.scaleStep = 1.0575;
        opt.scalePenalty = 0.9780;
        opt.scaleLR = 0.52;
        opt.wInfluence = 0.2625;
        opt.zLR = 0.005;
        
        opt.net_base = '/home/qi/projects/MDP_LOMOCF_Qi/CFNet/pretrained/networks/';
        opt.stats = '/home/qi/projects/MDP_LOMOCF_Qi/CFNet/data/ILSVRC2015.stats.mat';  
        
    case 'lomo'
        
        addpath(genpath('/home/qi/projects/MDP_LOMOCF_Qi/LOMO_XQDA/bin'));
        addpath(genpath('/home/qi/projects/MDP_LOMOCF_Qi/LOMO_XQDA/code'));
        
        opt.max_iter = 30;   %right
        opt.max_count = 5;
        
%         opt.max_iter = 500;    %wrong to record for ICPR2018
%         opt.max_count = 7;
        
        opt.numScales = 3;     %..
        opt.blockSize = 10;    %..
        opt.blockStep = 5;
        opt.hsvBins = [8,8,8];
        opt.tau = 0.3;
        opt.R = [3, 5];        %..
        opt.numPoints = 4;
        
    case 'SCT'
        
        addpath(genpath('/home/qi/projects/MDP_LOMOCF_Qi/SCT/KCF'));
        addpath(genpath('/home/qi/projects/MDP_LOMOCF_Qi/SCT/strong'));
        addpath(genpath('/home/qi/projects/MDP_LOMOCF_Qi/SCT/PiotrDollarToolbox'));
        
        opt.max_iter = 50;
        opt.max_count = 8;
        
        % Feature & Kernel parameters setting
        opt.interp_factor = 0.02;
        opt.kernel.sigma = 0.5;

        opt.kernel.poly_a = 1;
        opt.kernel.poly_b = 9;

        opt.features.gray = false;
        opt.features.hog = true;
        opt.features.hog_orientations = 9;
        opt.cell_size = 4;

        % KCF parameters setting
        opt.padding = 1.5;  %extra area surrounding the target
        opt.lambda = 1e-4;  %regularization
        opt.output_sigma_factor = 0.1;  %spatial bandwidth (proportional to target)

        % Attention map parameters setting
        opt.Nfo = 10;
        opt.boundary_ratio = 1/3;
        opt.salWeight = [0.3 0.3];
        opt.bSal = [1 1];

        % multiple module trackers type initialization
        opt.filterPool(1).kernelType = 'gaussian';
        opt.filterPool(1).featureType = 'color';
        opt.filterPool(2).kernelType = 'polynomial';
        opt.filterPool(2).featureType = 'color';
        opt.filterPool(3).kernelType = 'gaussian';
        opt.filterPool(3).featureType = 'hog';
        opt.filterPool(4).kernelType = 'polynomial';
        opt.filterPool(4).featureType = 'hog';
        
    case 'ADNet'
%         addpath('test/');
%         addpath(genpath('utils/'));
        
        addpath(genpath('/home/qi/projects/MDP_LOMOCF_Qi/ADNet/test/'));
        addpath(genpath('/home/qi/projects/MDP_LOMOCF_Qi/ADNet/utils/'));
        
        init_settings;
        
        run(matconvnet_path);
        
%         % net
%         load('ADNet/models/net_rl.mat');
        
        opt.visualize = true;
        opt.printscreen = true;
        
        rng(1004);
        
        opt.max_iter = 30;
        opt.max_count = 5;
        
    case 'lomocf'        
        % parameters for kcf tracking   % ------add
        opt.max_iter = 500;     % max iterations in total        %------change
        opt.max_count = 7;       % max iterations per sequence  % -----change
        
        kernel.type = 'gaussian';
        kernel.sigma = 0.5;       %gaussian kernel bandwidth

        kernel.poly_a = 1;        %polynomial kernel additive term
        kernel.poly_b = 9;        %polynomial kernel exponent

        features.gray = false;
        features.hog = true;
        features.hog_orientations = 9;

        opt.cell_size = 4;
        opt.features = features;
        opt.kernel = kernel;
        opt.padding = 1.5;
        opt.output_sigma_factor = 0.1;
        opt.interp_factor_1 = 0.02;
        opt.interp_factor_2 = 0.1;
        opt.lambda = 1e-4;
        opt.template_sz = 96;
        
        % lomo
        addpath(genpath('/home/qi/projects/MDP_LOMOCF_Qi/LOMO_XQDA/bin'));
        addpath(genpath('/home/qi/projects/MDP_LOMOCF_Qi/LOMO_XQDA/code'));
                
        opt.numScales = 3;     %..
        opt.blockSize = 10;    %..
        opt.blockStep = 5;
        opt.hsvBins = [8,8,8];
        opt.tau = 0.3;
        opt.R = [3, 5];        %..
        opt.numPoints = 4;
        
    case 'ori'
%         opt.max_iter = 10000;     
%         opt.max_count = 10;

        opt.max_iter = 500;     
        opt.max_count = 7;
        
end