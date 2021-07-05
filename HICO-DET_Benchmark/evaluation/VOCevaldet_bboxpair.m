function [rec, prec, ap] = VOCevaldet_bboxpair(det_id, det_bb, det_conf, gt_bbox, min_overlap, cls, draw)

% load ground truth objects
npos = 0;
gt(length(gt_bbox)) = struct('bb', [], 'det', []);
for i = 1:length(gt_bbox)
    if ~isempty(gt_bbox{i})
        gt(i).bb = gt_bbox{i};
        gt(i).det = false(size(gt_bbox{i}, 1), 1);
        npos = npos + sum(size(gt_bbox{i},1));
    end
end

% sort detections by decreasing confidence
[~, si] = sort(-det_conf);
det_id = det_id(si);
det_bb = det_bb(si, :);

% assign detections to ground truth objects
nd = length(det_conf);
tp = zeros(nd, 1);
fp = zeros(nd, 1);

for d = 1:nd
    % get gt id
    i = det_id(d);
    % get det box
    bb_1 = det_bb(d, 1:4);
    bb_2 = det_bb(d, 5:8);
    % set ov_max
    ov_max = -inf;
    for j = 1:size(gt(i).bb, 1)
        % get gt box
        bbgt_1 = gt(i).bb(j, 1:4);
        bbgt_2 = gt(i).bb(j, 5:8);
        % compare box 1
        bi_1 = [max(bb_1(1),bbgt_1(1)); max(bb_1(2),bbgt_1(2)); ...
            min(bb_1(3),bbgt_1(3)); min(bb_1(4),bbgt_1(4))];
        iw_1 = bi_1(3)-bi_1(1)+1;
        ih_1 = bi_1(4)-bi_1(2)+1;
        if iw_1 > 0 && ih_1 > 0
            % compute overlap as area of intersection / area of union
            ua_1 = (bb_1(3)-bb_1(1)+1)*(bb_1(4)-bb_1(2)+1) + ...
                (bbgt_1(3)-bbgt_1(1)+1)*(bbgt_1(4)-bbgt_1(2)+1) - ...
                iw_1*ih_1;
            ov_1 = iw_1*ih_1/ua_1;
        else
            ov_1 = 0;
        end
        % compare box 2
        bi_2 = [max(bb_2(1),bbgt_2(1)); max(bb_2(2),bbgt_2(2)); ...
            min(bb_2(3),bbgt_2(3)); min(bb_2(4),bbgt_2(4))];
        iw_2 = bi_2(3)-bi_2(1)+1;
        ih_2 = bi_2(4)-bi_2(2)+1;
        if iw_2 > 0 && ih_2 > 0
            % compute overlap as area of intersection / area of union
            ua_2 = (bb_2(3)-bb_2(1)+1)*(bb_2(4)-bb_2(2)+1) + ...
                (bbgt_2(3)-bbgt_2(1)+1)*(bbgt_2(4)-bbgt_2(2)+1) - ...
                iw_2*ih_2;
            ov_2 = iw_2*ih_2/ua_2;
        else
            ov_2 = 0;
        end
        % get minimum
        min_ov = min(ov_1, ov_2);
        % update ov_max & j_max
        if min_ov > ov_max
            ov_max = min_ov;
            j_max = j;
        end
    end
    % assign detection as true positive/don't care/false positive
    if ov_max >= min_overlap
        if ~gt(i).det(j_max)
            tp(d) = 1;             % true positive
            gt(i).det(j_max) = true;
        else
            fp(d) = 1;             % false positive (multiple detection)
        end
    else
        fp(d) = 1;                 % false positive
    end
end

% compute precision/recall
fp = cumsum(fp);
tp = cumsum(tp);
rec = tp/npos;
prec = tp./(fp+tp);

% compute average precision
ap = 0;
for t = 0:0.1:1
    p = max(prec(rec >= t));
    if isempty(p)
        p = 0;
    end
    ap = ap + p/11;
end

if draw
    % plot precision/recall
    plot(rec, prec, '-');
    grid;
    xlabel 'recall'
    ylabel 'precision'
    title(sprintf('class: %s, AP = %.3f', cls, ap));
end
