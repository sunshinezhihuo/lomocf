% --------------------------------------------------------
% MDP Tracking
% Copyright (c) 2015 CVGL Stanford
% Licensed under The MIT License [see LICENSE for details]
% Written by Yu Xiang
% --------------------------------------------------------
function opt = globals()

opt.root = pwd;

% path for MOT benchmark
% mot_paths = {'/capri5/Projects/Multitarget_Tracking/MOTbenchmark', ...
%     '/scail/scratch/u/yuxiang/MOTbenchmark'};
% mot_paths={'/home/djr/tracking/dataset/MOTbenchmark'};

% Load Datasets
mot_paths = {'/home/qi/projects/MDP_LOMOCF_Qi/MOTbenchmark'};

for i = 1:numel(mot_paths)
    if exist(mot_paths{i}, 'dir') == 0
        mkdir(mot_paths{i});
    end
    
    if exist(mot_paths{i}, 'dir')
        opt.mot = mot_paths{i};
        break;
    end
end
% choose a method
% '2DMOT2015'  MOT16

opt.mot2d = '2DMOT2015';
% opt.mot2d = 'MOT16';

opt.results = 'results';
opt.myhandinresults = 'myhandinresults';
opt.results_kitti = 'results_kitti';

switch opt.mot2d 
    case '2DMOT2015'
    opt.mot2d_train_seqs = {'TUD-Stadtmitte', 'TUD-Campus', 'PETS09-S2L1', ...
        'ETH-Bahnhof', 'ETH-Sunnyday', 'ETH-Pedcross2', 'ADL-Rundle-6', ...
        'ADL-Rundle-8', 'KITTI-13', 'KITTI-17', 'Venice-2'};
    opt.mot2d_train_nums = [179, 71, 795, 1000, 354, 837, 525, 654, 340, 145, 600];

    opt.mot2d_test_seqs = {'TUD-Crossing', 'PETS09-S2L2', 'ETH-Jelmoli', ...
        'ETH-Linthescher', 'ETH-Crossing', 'AVG-TownCentre', 'ADL-Rundle-1', ...
        'ADL-Rundle-3', 'KITTI-16', 'KITTI-19', 'Venice-1'};
    opt.mot2d_test_nums = [201, 436, 440, 1194, 219, 450, 500, 625, 209, 1059, 450];
    case 'MOT16'
%% train
% MOT16-02
% MOT16-04
% MOT16-05
% MOT16-09
% MOT16-10
% MOT16-11
% MOT16-13
    opt.mot2d_train_seqs = {'MOT16-02', 'MOT16-04', 'MOT16-05', ...
        'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13'};
    opt.mot2d_train_nums = [600, 1050, 837, 525, 654, 900, 750];

    opt.mot2d_test_seqs = {'MOT16-01', 'MOT16-03', 'MOT16-06', ...
        'MOT16-07', 'MOT16-08', 'MOT16-12', 'MOT16-14'};
    opt.mot2d_test_nums = [450, 1500, 1194, 500, 625, 900, 750];
end

% % path for KITTI tracking dataset
% kitti_paths = {'/home/qi/tracking/dataset/KITTI'};
% for i = 1:numel(kitti_paths)
%     if exist(kitti_paths{i}, 'dir')
%         opt.kitti = kitti_paths{i};
%         break;
%     end
% end
% 
% opt.kitti_train_seqs = {'0000', '0001', '0002', '0003', '0004', '0005', ...
%     '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', ...
%     '0015', '0016', '0017', '0018', '0019', '0020'};
% opt.kitti_train_nums = [154, 447, 233, 144, 314, 297, 270, 800, 390, 803, 294, ...
%     373, 78, 340, 106, 376, 209, 145, 339, 1059, 837];
% 
% opt.kitti_test_seqs = {'0000', '0001', '0002', '0003', '0004', '0005', ...
%     '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', ...
%     '0015', '0016', '0017', '0018', '0019', '0020', '0021', '0022', ...
%     '0023', '0024', '0025', '0026', '0027', '0028'};
% opt.kitti_test_nums = [465, 147, 243, 257, 421, 809, 114, 215, 165, 349, 1176, ...
%     774, 694, 152, 850, 701, 510, 305, 180, 404, 173, 203, 436, 430, 316, 176, ...
%     170, 85, 175];
% opt.kitti_types = {'Car', 'Pedestrian', 'Cyclist'};

addpath(fullfile(opt.mot, 'devkit', 'utils'));
% addpath(fullfile(opt.kitti, 'devkit', 'matlab'));
addpath([opt.root '/3rd_party/libsvm-3.20/matlab']);
addpath([opt.root '/3rd_party/Hungarian']);

%----------------------------------------------------------------------------CF
addpath([opt.root '/BACF']);
addpath([opt.root '/CFNet']);
addpath([opt.root '/LOMO_XQDA']);
addpath([opt.root '/SCT']);
addpath([opt.root '/ADNet']);

if exist(opt.results, 'dir') == 0
    mkdir(opt.results);
end

if exist(opt.myhandinresults, 'dir') == 0
    mkdir(opt.myhandinresults);
end

if exist(opt.results_kitti, 'dir') == 0
    mkdir(opt.results_kitti);
end

% tracking parameters
opt.num = 10;                 % number of templates in tracker (default 10)
opt.fb_factor = 30;           % normalization factor for forward-backward error in optical flow
opt.threshold_ratio = 0.6;    % aspect ratio threshold in target association
opt.threshold_dis = 3;        % distance threshold in target association, multiple of the width of target
opt.threshold_box = 0.8;      % bounding box overlap threshold in tracked state
opt.std_box = [30 60];        % [width height] of the stanford box in computing flow
opt.margin_box = [5, 2];      % [width height] of the margin in computing flow
opt.enlarge_box = [5, 3];     % enlarge the box before computing flow
opt.level_track = 1;          % LK level in association
opt.level =  1;               % LK level in association
opt.max_ratio = 0.9;          % min allowed ratio in LK
opt.min_vnorm = 0.2;          % min allowed velocity norm in LK
opt.overlap_box = 0.5;        % overlap with detection in LK
opt.patchsize = [24 12];      % patch size for target appearance
opt.weight_tracking = 1;      % weight for tracking box in tracked state
opt.weight_detection = 1;      % weight for detection box in tracked state     %----add
opt.weight_association = 1;   % weight for tracking box in lost state
opt.overlap_suppress1 = 0.5;   % overlap for suppressing detections with tracked objects   % ----add
opt.overlap_suppress2 = 0.5;   % overlap for suppressing detections with tracked objects   % ----add

% parameters for generating training data
opt.overlap_occ = 0.7;
opt.overlap_pos = 0.5;
opt.overlap_neg = 0.2;
opt.overlap_sup = 0.7;      % suppress target used in testing only

% % training parameters     % Original
% opt.max_iter = 10000;     % max iterations in total
% opt.max_count = 10;       % max iterations per sequence
% opt.max_pass = 2;

% training parameters   I put it in choosemethod.m  
% opt.max_iter = 30;     % max iterations in total        %------change 
% opt.max_count = 5;       % max iterations per sequence  % -----change

opt.max_pass = 2;

% parameters to transite to inactive
opt.max_occlusion = 50;
opt.exit_threshold = 0.95;
opt.tracked = 5;

choosemethod;


