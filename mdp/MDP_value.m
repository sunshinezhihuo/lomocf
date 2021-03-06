% --------------------------------------------------------
% MDP Tracking
% Copyright (c) 2015 CVGL Stanford
% Licensed under The MIT License [see LICENSE for details]
% Written by Yu Xiang
% --------------------------------------------------------
%
% MDP value function
function [tracker, qscore, f] = MDP_value(tracker, frame_id, dres_image, dres_det, index_det, opt)

% tracked, decide to tracked or occluded
if tracker.state == 2
    % extract features with LK tracking
    [tracker, f] = MDP_feature_tracked(frame_id, dres_image, dres_det, tracker,opt);   % ------cf add opt
    
    % build the dres structure
    if bb_isdef(tracker.bb)
        dres_one.fr = frame_id;
        dres_one.id = tracker.target_id;
        dres_one.x = tracker.bb(1);
        dres_one.y = tracker.bb(2);
        dres_one.w = tracker.bb(3) - tracker.bb(1);
        dres_one.h = tracker.bb(4) - tracker.bb(2);
        dres_one.r = 1;
    else
        dres_one = sub(tracker.dres, numel(tracker.dres.fr));
        dres_one.fr = frame_id;
        dres_one.id = tracker.target_id;
    end
    
    if isfield(tracker.dres, 'type')
        dres_one.type = tracker.dres.type{1};
    end
    
    % compute qscore
    qscore = 0;
    if f(1) == 1 && f(2) > tracker.threshold_box
        label = 1;
    else
        label = -1;
    end

    % make a decision
    if label > 0
        tracker.state = 2;
        dres_one.state = 2;
        tracker.dres = concatenate_dres(tracker.dres, dres_one);
        % update LK tracker
        switch opt.method
            case 'KCF'
                tracker = LK_update(frame_id, tracker, dres_image.Igray{frame_id}, dres_det, 0);    %---------------------cf
            case 'BACF'
                tracker = LK_update_bacf(frame_id, tracker, dres_image.Igray{frame_id},dres_image.I{frame_id}, dres_det, 0);        % -----------------------cf
            case 'CFNet'
                tracker = LK_update_cfnet(frame_id, tracker, dres_image.Igray{frame_id},dres_image.I{frame_id}, dres_det, 0);        % -----------------------cf
            case 'lomo'
                tracker = LK_update_lomo(frame_id, tracker, dres_image.Igray{frame_id},dres_image.I{frame_id}, dres_det, 0);
            case 'SCT'
                tracker = LK_update_SCT(frame_id, tracker, dres_image.Igray{frame_id},dres_image.I{frame_id}, dres_det, 0);
            case 'ADNet'
                tracker = LK_update_ADNet(frame_id, tracker,opt, dres_image.Igray{frame_id},dres_image.I{frame_id}, dres_det, 0);                 
            case 'lomocf'
                % kcf
                tracker = LK_update(frame_id, tracker, dres_image.Igray{frame_id}, dres_det, 0);
            case 'ori'
                tracker = LK_update_ori(frame_id, tracker, dres_image.Igray{frame_id}, dres_det, 0);
%                 tracker = LK_update_lomo(frame_id, tracker, dres_image.Igray{frame_id},dres_image.I{frame_id}, dres_det, 0);

                
        end
        
    else
        tracker.state = 3;
        dres_one.state = 3;
        tracker.dres = concatenate_dres(tracker.dres, dres_one);        
    end
    tracker.prev_state = 2;

% occluded, decide to tracked or occluded
elseif tracker.state == 3

    % association
    if isempty(index_det) == 1
        qscore = 0;
        label = -1;
        f = [];
    else
        % extract features with LK association
        dres = sub(dres_det, index_det);
        [features, flag] = MDP_feature_occluded(frame_id, dres_image, dres, tracker,opt);    %------------------cf

        m = size(features, 1);
        labels = -1 * ones(m, 1);
        [labels, ~, probs] = svmpredict(labels, features, tracker.w_occluded, '-b 1 -q');

        probs(flag == 0, 1) = 0;
        probs(flag == 0, 2) = 1;
        labels(flag == 0) = -1;

        [qscore, ind] = max(probs(:,1));
        label = labels(ind);
        f = features(ind,:);

        dres_one = sub(dres_det, index_det(ind));
        tracker = LK_associate(frame_id, dres_image, dres_one, tracker,opt);       %-------------------cf
    end

    % make a decision
    tracker.prev_state = tracker.state;
    if label > 0
        % association
        tracker.state = 2;
        % build the dres structure
        dres_one = [];
        dres_one.fr = frame_id;
        dres_one.id = tracker.target_id;
        dres_one.x = tracker.bb(1);
        dres_one.y = tracker.bb(2);
        dres_one.w = tracker.bb(3) - tracker.bb(1);
        dres_one.h = tracker.bb(4) - tracker.bb(2);
        dres_one.r = 1;
        dres_one.state = 2;
        
        if isfield(tracker.dres, 'type')
            dres_one.type = tracker.dres.type{1};
        end        
        
        if tracker.dres.fr(end) == frame_id
            dres = tracker.dres;
            index = 1:numel(dres.fr)-1;
            tracker.dres = sub(dres, index);            
        end
        tracker.dres = interpolate_dres(tracker.dres, dres_one);
        % update LK tracker
                                                %----------------cf
        switch opt.method
            case 'KCF'
                tracker = LK_update(frame_id, tracker, dres_image.Igray{frame_id}, dres_det, 1);      
            case 'BACF'
                tracker = LK_update_bacf(frame_id, tracker, dres_image.Igray{frame_id}, dres_image.I{frame_id}, dres_det, 1);
            case 'CFNet'
                tracker = LK_update_cfnet(frame_id, tracker, dres_image.Igray{frame_id}, dres_image.I{frame_id}, dres_det, 1);    
            case 'lomo'
                tracker = LK_update_lomo(frame_id, tracker, dres_image.Igray{frame_id}, dres_image.I{frame_id}, dres_det, 1);
            case 'SCT'
                tracker = LK_update_SCT(frame_id, tracker, dres_image.Igray{frame_id}, dres_image.I{frame_id}, dres_det, 1);
            case 'ADNet'
                tracker = LK_update_ADNet(frame_id, tracker,opt, dres_image.Igray{frame_id},dres_image.I{frame_id}, dres_det, 0);
            case 'lomocf'
                % kcf
                tracker = LK_update(frame_id, tracker, dres_image.Igray{frame_id}, dres_det, 1);
            case 'ori'
                tracker = LK_update_ori(frame_id, tracker, dres_image.Igray{frame_id}, dres_det, 0);
%                 tracker = LK_update_lomo(frame_id, tracker, dres_image.Igray{frame_id},dres_image.I{frame_id}, dres_det, 0);

        end
    else
        % no association
        tracker.state = 3;
        dres_one = sub(tracker.dres, numel(tracker.dres.fr));
        dres_one.fr = frame_id;
        dres_one.id = tracker.target_id;
        dres_one.state = 3;
        
        if tracker.dres.fr(end) == frame_id
            dres = tracker.dres;
            index = 1:numel(dres.fr)-1;
            tracker.dres = sub(dres, index);
        end        
        tracker.dres = concatenate_dres(tracker.dres, dres_one);          
    end
end

% if tracker.is_show
%     fprintf('qscore %.2f\n', qscore);
% end