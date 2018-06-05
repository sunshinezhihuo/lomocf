% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.
% 
% TLD is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% TLD is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with TLD.  If not, see <http://www.gnu.org/licenses/>.

% modified by Yu Xiang

function tracker = LK_tracking(frame_id, dres_image, dres_det, tracker,opt)

% current frame + motion
J = dres_image.Igray{frame_id};
JColor = dres_image.I{frame_id};                              % -------------------for bacf color

% motion model original
ctrack = apply_motion_prediction(frame_id, tracker);
w = tracker.dres.w(end);
h = tracker.dres.h(end);
BB4 = [ctrack(1)-w/2; ctrack(2)-h/2; ctrack(1)+w/2; ctrack(2)+h/2];

%debug
% fprintf('motion track result BB4:%f,%f,%f,%f\n',BB4(1),BB4(2),BB4(3),BB4(4));
%KCF tracker
% [ BB3,tracker ] = kcf_predict( J, BB4,tracker);
% %debug
% tracker.kcf_bb = BB3;
%  fprintf('CF track result BB3:%03f,%03f,%03f,%03f\n',BB3(1),BB3(2),BB3(3),BB3(4));
%% Compute box 
switch opt.method 
    case 'KCF'
        [ BB5,tracker,appsim] = kcf_predict( J, BB4,tracker);        
        
        tracker.kcf_bb = BB5; 

%         BB3 = BB4;   % try 25.6,but -
        BB3 = BB5;
        
    case 'BACF'
%         % debug 
%         figure(4)
%         imshow(JColor);
%         rectangle('Position', [BB4(1) BB4(2) (BB4(3)-BB4(1)) (BB4(4)-BB4(2))], 'EdgeColor', 'g', 'LineWidth', 2, 'LineStyle', '-');
%         hold on;

        [ BB5,tracker,appsim] = bacf_predict( JColor, BB4,tracker,opt);
%         % debug 
%         figure(5)
%         imshow(JColor);
%         rectangle('Position', [BB5(1) BB5(2) (BB5(3)-BB5(1)) (BB5(4)-BB5(2))], 'EdgeColor', 'b', 'LineWidth', 2, 'LineStyle', '-');
%         hold on;
        
        %debug
        tracker.kcf_bb = BB5;
%          fprintf('CF track result BB3:%03f,%03f,%03f,%03f\n',BB5(1),BB5(2),BB5(3),BB5(4));
   
        BB3 = (BB4+BB5)/2;       
        
%         % debug 
%         figure(3)
%         imshow(JColor);
%         rectangle('Position', [BB3(1) BB3(2) (BB3(3)-BB3(1)) (BB3(4)-BB3(2))], 'EdgeColor', 'r', 'LineWidth', 2, 'LineStyle', '-');
%         hold on;
%         fprintf('bbox compute:%03f,%03f,%03f,%03f\n',BB3(1),BB3(2),BB3(3),BB3(4));
    
    case 'CFNet'
        
        [ BB5,tracker,appsim] = cfnet_predict( JColor, BB4,tracker,opt);    
        tracker.kcf_bb = BB5;
        BB3 = BB5;
        
    case 'lomo'
%         tracker.kcf_bb = BB4;
        BB3 = BB4;
        
    case 'SCT'
        [ BB5,tracker,appsim] = SCT_predict( JColor, BB4,tracker,opt);
        tracker.kcf_bb = BB5;
        BB3 = BB5;
         
% % debug 
%         figure(4)
%         imshow(JColor);
%         rectangle('Position', [BB4(1) BB4(2) (BB4(3)-BB4(1)) (BB4(4)-BB4(2))], 'EdgeColor', 'g', 'LineWidth', 2, 'LineStyle', '-');
%         hold on;
%         fprintf('original track BB4 result BB4:%03f,%03f,%03f,%03f\n',BB4(1),BB4(2),BB4(3),BB4(4));
%         figure(5)
%         imshow(JColor);
%         rectangle('Position', [BB5(1) BB5(2) (BB5(3)-BB5(1)) (BB5(4)-BB5(2))], 'EdgeColor', 'r', 'LineWidth', 2, 'LineStyle', '-');
%         hold on;
%         fprintf('original track BB5 result BB5:%03f,%03f,%03f,%03f\n',BB5(1),BB5(2),BB5(3),BB5(4));

    case 'ADNet'
        [ BB5,tracker,appsim] = ADNet_predict( J, BB4,tracker,frame_id,opt);
        
        tracker.kcf_bb = BB5;
        BB3 = BB5;
        
        
    case 'lomocf'
        
  
        [ BB5,tracker,appsim] = kcf_predict( J, BB4,tracker);        
        
        tracker.kcf_bb = BB5;        
        BB3 = BB4;   % try 25.6
        
    case 'ori'
        BB3 = BB4;

end
 
 %%
[J_crop, BB3_crop, bb_crop, s] = LK_crop_image_box(J, BB3, tracker);

num_det = numel(dres_det.x);
for i = 1:tracker.num
    BB1 = [tracker.x1(i); tracker.y1(i); tracker.x2(i); tracker.y2(i)];
    I_crop = tracker.Is{i};
    BB1_crop = tracker.BBs{i};
    
    % LK tracking
    [BB2, xFJ, flag, medFB, medNCC, medFB_left, medFB_right, medFB_up, medFB_down] = LK(I_crop, J_crop, ...
        BB1_crop, BB3_crop, tracker.margin_box, tracker.level_track);
    
    BB2 = bb_shift_absolute(BB2, [bb_crop(1) bb_crop(2)]);
    BB2 = [BB2(1)/s(1); BB2(2)/s(2); BB2(3)/s(1); BB2(4)/s(2)];

    ratio = (BB2(4)-BB2(2)) / (BB1(4)-BB1(2));
    ratio = min(ratio, 1/ratio);
    
    if isnan(medFB) || isnan(medFB_left) || isnan(medFB_right) || isnan(medFB_up) || isnan(medFB_down) ...
            || isnan(medNCC) || ~bb_isdef(BB2) || ratio < tracker.max_ratio
        medFB = inf;
        medFB_left = inf;
        medFB_right = inf;
        medFB_up = inf;
        medFB_down = inf;
        medNCC = 0;
        o = 0;
        score = 0;
        ind = 1;
        angle = -1;
        flag = 2;
        BB2 = [NaN; NaN; NaN; NaN];
    else
        % compute overlap
        dres.x = BB2(1);
        dres.y = BB2(2);
        dres.w = BB2(3) - BB2(1);
        dres.h = BB2(4) - BB2(2);
        if isempty(dres_det.fr) == 0
            overlap = calc_overlap(dres, 1, dres_det, 1:num_det);
            [o, ind] = max(overlap);
            score = dres_det.r(ind);
        else
            o = 0;
            score = -1;
            ind = 0;
        end
        
        % compute angle
        centerI = [(BB1(1)+BB1(3))/2 (BB1(2)+BB1(4))/2];
        centerJ = [(BB2(1)+BB2(3))/2 (BB2(2)+BB2(4))/2];
        v = compute_velocity(tracker);
        v_new = [centerJ(1)-centerI(1), centerJ(2)-centerI(2)] / double(frame_id - tracker.frame_ids(i));
        if norm(v) > tracker.min_vnorm && norm(v_new) > tracker.min_vnorm
            angle = dot(v, v_new) / (norm(v) * norm(v_new));
        else
            angle = 1;
        end        
    end
    
    tracker.bbs{i} = BB2;
    tracker.points{i} = xFJ;
    tracker.flags(i) = flag;
    tracker.medFBs(i) = medFB;
    tracker.medFBs_left(i) = medFB_left;
    tracker.medFBs_right(i) = medFB_right;
    tracker.medFBs_up(i) = medFB_up;
    tracker.medFBs_down(i) = medFB_down;
    tracker.medNCCs(i) = medNCC;
    tracker.overlaps(i) = o;
    tracker.scores(i) = score;
    tracker.indexes(i) = ind;
    tracker.angles(i) = angle;
    tracker.ratios(i) = ratio;
end

% combine tracking and detection results
% [~, ind] = min(tracker.medFBs);
ind = tracker.anchor;

switch opt.method 
    case 'KCF'
        
        if tracker.overlaps(ind) > tracker.overlap_box
            index = tracker.indexes(ind);
            bb_det = [dres_det.x(index); dres_det.y(index); ...
                dres_det.x(index)+dres_det.w(index); dres_det.y(index)+dres_det.h(index)];

        %tracker.bb = mean([repmat(tracker.bbs{ind}, 1, tracker.weight_tracking) bb_det], 2);   

            bb_track = mean([repmat(tracker.bbs{ind},1,1) repmat(tracker.kcf_bb,1,1)],2);           % -------------------cf    
            tracker.bb = mean([repmat(bb_track, 1, tracker.weight_tracking) ...
                repmat(bb_det, 1, tracker.weight_detection)], 2);

            %debug
            lk_bb = tracker.bbs{ind};
            tracker.lk_bb = lk_bb;
        %     fprintf('detection result:%d,%d,%d,%d\n',bb_det(1),bb_det(2),bb_det(3),bb_det(4));
        %     fprintf('lk track result:%03f,%03f,%03f,%03f\n',lk_bb(1),lk_bb(2),lk_bb(3),lk_bb(4));
        else
            tracker.bb = tracker.bbs{ind};
        end
    case 'BACF'
        if tracker.overlaps(ind) > tracker.overlap_box
            index = tracker.indexes(ind);
            bb_det = [dres_det.x(index); dres_det.y(index); ...
                dres_det.x(index)+dres_det.w(index); dres_det.y(index)+dres_det.h(index)];

        %tracker.bb = mean([repmat(tracker.bbs{ind}, 1, tracker.weight_tracking) bb_det], 2);   

            bb_track = mean([repmat(tracker.bbs{ind},1,1) repmat(tracker.kcf_bb,1,1)],2);           % -------------------cf    
            tracker.bb = mean([repmat(bb_track, 1, tracker.weight_tracking) ...
                repmat(bb_det, 1, tracker.weight_detection)], 2);

            %debug
            lk_bb = tracker.bbs{ind};
            tracker.lk_bb = lk_bb;
        %     fprintf('detection result:%d,%d,%d,%d\n',bb_det(1),bb_det(2),bb_det(3),bb_det(4));
        %     fprintf('lk track result:%03f,%03f,%03f,%03f\n',lk_bb(1),lk_bb(2),lk_bb(3),lk_bb(4));
        else
            tracker.bb = tracker.bbs{ind};
        end
    case 'CFNet'
        if tracker.overlaps(ind) > tracker.overlap_box
            index = tracker.indexes(ind);
            bb_det = [dres_det.x(index); dres_det.y(index); ...
                dres_det.x(index)+dres_det.w(index); dres_det.y(index)+dres_det.h(index)];

        %tracker.bb = mean([repmat(tracker.bbs{ind}, 1, tracker.weight_tracking) bb_det], 2);   

            bb_track = mean([repmat(tracker.bbs{ind},1,1) repmat(tracker.kcf_bb,1,1)],2);           % -------------------cf    
            tracker.bb = mean([repmat(bb_track, 1, tracker.weight_tracking) ...
                repmat(bb_det, 1, tracker.weight_detection)], 2);

            %debug
            lk_bb = tracker.bbs{ind};
            tracker.lk_bb = lk_bb;
        %     fprintf('detection result:%d,%d,%d,%d\n',bb_det(1),bb_det(2),bb_det(3),bb_det(4));
        %     fprintf('lk track result:%03f,%03f,%03f,%03f\n',lk_bb(1),lk_bb(2),lk_bb(3),lk_bb(4));
        else
            tracker.bb = tracker.bbs{ind};
        end
        
    case 'lomo'
        if tracker.overlaps(ind) > tracker.overlap_box
            index = tracker.indexes(ind);
            bb_det = [dres_det.x(index); dres_det.y(index); ...
            dres_det.x(index)+dres_det.w(index); dres_det.y(index)+dres_det.h(index)];
            tracker.bb = mean([repmat(tracker.bbs{ind}, 1, tracker.weight_tracking) bb_det], 2);
        else
            tracker.bb = tracker.bbs{ind};
        end
    case 'SCT'
        if tracker.overlaps(ind) > tracker.overlap_box
            index = tracker.indexes(ind);
            bb_det = [dres_det.x(index); dres_det.y(index); ...
                dres_det.x(index)+dres_det.w(index); dres_det.y(index)+dres_det.h(index)];

        %tracker.bb = mean([repmat(tracker.bbs{ind}, 1, tracker.weight_tracking) bb_det], 2);   

            bb_track = mean([repmat(tracker.bbs{ind},1,1) repmat(tracker.kcf_bb,1,1)],2);           % -------------------cf    
            tracker.bb = mean([repmat(bb_track, 1, tracker.weight_tracking) ...
                repmat(bb_det, 1, tracker.weight_detection)], 2);

            %debug
            lk_bb = tracker.bbs{ind};
            tracker.lk_bb = lk_bb;
        %     fprintf('detection result:%d,%d,%d,%d\n',bb_det(1),bb_det(2),bb_det(3),bb_det(4));
        %     fprintf('lk track result:%03f,%03f,%03f,%03f\n',lk_bb(1),lk_bb(2),lk_bb(3),lk_bb(4));
        else
            tracker.bb = tracker.bbs{ind};
        end
        
    case 'ADNet'
        
        if tracker.overlaps(ind) > tracker.overlap_box
            index = tracker.indexes(ind);
            bb_det = [dres_det.x(index); dres_det.y(index); ...
                dres_det.x(index)+dres_det.w(index); dres_det.y(index)+dres_det.h(index)];

        %tracker.bb = mean([repmat(tracker.bbs{ind}, 1, tracker.weight_tracking) bb_det], 2);   

            bb_track = mean([repmat(tracker.bbs{ind},1,1) repmat(tracker.kcf_bb,1,1)],2);           % -------------------cf    
            tracker.bb = mean([repmat(bb_track, 1, tracker.weight_tracking) ...
                repmat(bb_det, 1, tracker.weight_detection)], 2);

            %debug
            lk_bb = tracker.bbs{ind};
            tracker.lk_bb = lk_bb;
        %     fprintf('detection result:%d,%d,%d,%d\n',bb_det(1),bb_det(2),bb_det(3),bb_det(4));
        %     fprintf('lk track result:%03f,%03f,%03f,%03f\n',lk_bb(1),lk_bb(2),lk_bb(3),lk_bb(4));
        else
            tracker.bb = tracker.bbs{ind};
        end
        
    case 'lomocf'
%         % Without using SOT 24.6
%         if tracker.overlaps(ind) > tracker.overlap_box
%             index = tracker.indexes(ind);
%             bb_det = [dres_det.x(index); dres_det.y(index); ...
%                 dres_det.x(index)+dres_det.w(index); dres_det.y(index)+dres_det.h(index)];
%             tracker.bb = mean([repmat(tracker.bbs{ind}, 1, tracker.weight_tracking) bb_det], 2);
%             
%             lk_bb = tracker.bbs{ind};
%             tracker.lk_bb = lk_bb;
%         else
%             tracker.bb = tracker.bbs{ind};
%         end
        
        % using SOT 27.0
        if tracker.overlaps(ind) > tracker.overlap_box
            index = tracker.indexes(ind);
            bb_det = [dres_det.x(index); dres_det.y(index); ...
                dres_det.x(index)+dres_det.w(index); dres_det.y(index)+dres_det.h(index)];     

            bb_track = mean([repmat(tracker.bbs{ind},1,1) repmat(tracker.kcf_bb,1,1)],2);           % -----right------lomocf    
            tracker.bb = mean([repmat(bb_track, 1, tracker.weight_tracking) ...
                repmat(bb_det, 1, tracker.weight_detection)], 2);
                                        
            lk_bb = tracker.bbs{ind};
            tracker.lk_bb = lk_bb;
        
        else
            tracker.bb = tracker.bbs{ind};
        end
        
        
        
        %kcf       	
% %         if tracker.overlaps(ind) > tracker.overlap_box
% %             index = tracker.indexes(ind);
% %             bb_det = [dres_det.x(index); dres_det.y(index); ...
% %                 dres_det.x(index)+dres_det.w(index); dres_det.y(index)+dres_det.h(index)];     
% % 
% %             bb_track = mean([repmat(tracker.bbs{ind},1,1) repmat(tracker.kcf_bb,1,1)],2);           % -------------------cf    
% %             tracker.bb1 = mean([repmat(bb_track, 1, tracker.weight_tracking) ...
% %                 repmat(bb_det, 1, tracker.weight_detection)], 2);
% %                                         
% %             lk_bb = tracker.bbs{ind};
% %             tracker.lk_bb = lk_bb;
% %         
% %         else
% %             tracker.bb1 = tracker.bbs{ind};
% %         end
% %        %lomo
% %        if tracker.overlaps(ind) > tracker.overlap_box
% %             index = tracker.indexes(ind);
% %             bb_det = [dres_det.x(index); dres_det.y(index); ...
% %             dres_det.x(index)+dres_det.w(index); dres_det.y(index)+dres_det.h(index)];
% %             tracker.bb2 = mean([repmat(tracker.bbs{ind}, 1, tracker.weight_tracking) bb_det], 2);
% %         else
% %             tracker.bb2 = tracker.bbs{ind};
% %        end
% % 
% %        %lomocf mean
% %        tracker.bb = mean([repmat(tracker.bb1, 1, tracker.weight_tracking) ...
% %                 repmat(tracker.bb2, 1, tracker.weight_tracking)], 2);
                    
        % debug
        % detections
%         figure(11)
%         imshow(JColor);
%         rectangle('Position', [bb_det(1) bb_det(2) (bb_det(3)-bb_det(1)) (bb_det(4)-bb_det(2))], 'EdgeColor', 'r', 'LineWidth', 2, 'LineStyle', '-');
%         hold on;
%         % sot
%         figure(12)
%         imshow(JColor);
%         rectangle('Position', [tracker.kcf_bb(1) tracker.kcf_bb(2) (tracker.kcf_bb(3)-tracker.kcf_bb(1)) (tracker.kcf_bb(4)-tracker.kcf_bb(2))], 'EdgeColor', 'g', 'LineWidth', 2, 'LineStyle', '-');
%         hold on;
%         
%         % original
%         figure(13)
%         imshow(JColor);
%         rectangle('Position', [tracker.bb2(1) tracker.bb2(2) (tracker.bb2(3)-tracker.bb2(1)) (tracker.bb2(4)-tracker.bb2(2))], 'EdgeColor', 'b', 'LineWidth', 2, 'LineStyle', '-');
%         hold on;
%         
%         % final
%         figure(14)
%         imshow(JColor);
%         rectangle('Position', [tracker.bb(1) tracker.bb(2) (tracker.bb(3)-tracker.bb(1)) (tracker.bb(4)-tracker.bb(2))], 'EdgeColor', 'yellow', 'LineWidth', 2, 'LineStyle', '-');
%         hold on;

        % Qi debug to say the theory
%         figure(15)
%         imshow(JColor);
%         %Detection red
%         rectangle('Position', [bb_det(1) bb_det(2) (bb_det(3)-bb_det(1)) (bb_det(4)-bb_det(2))], 'EdgeColor', 'red', 'LineWidth', 2, 'LineStyle', '-');
%         %SOT green
%         rectangle('Position', [tracker.kcf_bb(1) tracker.kcf_bb(2) (tracker.kcf_bb(3)-tracker.kcf_bb(1)) (tracker.kcf_bb(4)-tracker.kcf_bb(2))], 'EdgeColor', 'green', 'LineWidth', 2, 'LineStyle', '-');        
%         %LK tracking blue
%         rectangle('Position', [lk_bb(1) lk_bb(2) (lk_bb(3)-lk_bb(1)) (lk_bb(4)-lk_bb(2))], 'EdgeColor', 'blue', 'LineWidth', 2, 'LineStyle', '-');
%         %Our method yellow
%         rectangle('Position', [tracker.bb(1) tracker.bb(2) (tracker.bb(3)-tracker.bb(1)) (tracker.bb(4)-tracker.bb(2))], 'EdgeColor', 'yellow', 'LineWidth', 2, 'LineStyle', '-');
%         
%         t = text(30, 30, int2str(frame_id),'color','yellow');
%         t(1).FontSize = 16;
%         hold off;
        
    case 'ori'
        if tracker.overlaps(ind) > tracker.overlap_box
            index = tracker.indexes(ind);
            bb_det = [dres_det.x(index); dres_det.y(index); ...
            dres_det.x(index)+dres_det.w(index); dres_det.y(index)+dres_det.h(index)];
            tracker.bb = mean([repmat(tracker.bbs{ind}, 1, tracker.weight_tracking) bb_det], 2);
        else
            tracker.bb = tracker.bbs{ind};
        end
        
end

% compute pattern similarity
if bb_isdef(tracker.bb)
    pattern = generate_pattern(dres_image.Igray{frame_id}, tracker.bb, tracker.patchsize);
    nccs = distance(pattern, tracker.patterns, 1); % measure NCC to positive examples
    tracker.nccs = nccs';
else
    tracker.nccs = zeros(tracker.num, 1);
end    

if tracker.is_show
    fprintf('\ntarget %d: frame ids ', tracker.target_id);
    for i = 1:tracker.num
        fprintf('%d ', tracker.frame_ids(i))
    end
    fprintf('\n');    
    fprintf('target %d: medFB ', tracker.target_id);
    for i = 1:tracker.num
        fprintf('%.2f ', tracker.medFBs(i))
    end
    fprintf('\n');
    
    fprintf('target %d: medFB left ', tracker.target_id);
    for i = 1:tracker.num
        fprintf('%.2f ', tracker.medFBs_left(i))
    end
    fprintf('\n');
    
    fprintf('target %d: medFB right ', tracker.target_id);
    for i = 1:tracker.num
        fprintf('%.2f ', tracker.medFBs_right(i))
    end
    fprintf('\n');
    
    fprintf('target %d: medFB up ', tracker.target_id);
    for i = 1:tracker.num
        fprintf('%.2f ', tracker.medFBs_up(i))
    end
    fprintf('\n');
    
    fprintf('target %d: medFB down ', tracker.target_id);
    for i = 1:tracker.num
        fprintf('%.2f ', tracker.medFBs_down(i))
    end
    fprintf('\n');       
    
    fprintf('target %d: medNCC ', tracker.target_id);
    for i = 1:tracker.num
        fprintf('%.2f ', tracker.medNCCs(i))
    end
    fprintf('\n');
    
    fprintf('target %d: overlap ', tracker.target_id);
    for i = 1:tracker.num
        fprintf('%.2f ', tracker.overlaps(i))
    end
    fprintf('\n');
    fprintf('target %d: detection score ', tracker.target_id);
    for i = 1:tracker.num
        fprintf('%.2f ', tracker.scores(i))
    end
    fprintf('\n');
    fprintf('target %d: flag ', tracker.target_id);
    for i = 1:tracker.num
        fprintf('%d ', tracker.flags(i))
    end
    fprintf('\n');
    fprintf('target %d: angle ', tracker.target_id);
    for i = 1:tracker.num
        fprintf('%.2f ', tracker.angles(i))
    end
    fprintf('\n');
    fprintf('target %d: ncc ', tracker.target_id);
    for i = 1:tracker.num
        fprintf('%.2f ', tracker.nccs(i))
    end
    fprintf('\n\n');
    fprintf('target %d: bb overlaps ', tracker.target_id);
    for i = 1:tracker.num
        fprintf('%.2f ', tracker.bb_overlaps(i))
    end
    fprintf('\n\n');

    if tracker.flags(ind) == 2
        fprintf('target %d: bounding box out of image\n', tracker.target_id);
    elseif tracker.flags(ind) == 3
        fprintf('target %d: too unstable predictions\n', tracker.target_id);
    end
end