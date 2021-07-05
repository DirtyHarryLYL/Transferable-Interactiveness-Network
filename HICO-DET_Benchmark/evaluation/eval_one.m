
config;

min_overlap = 0.5;

% assertions
assert(ismember(score_blob,{'n/a','h','o','p'}) == 1);

% set detection root
det_root = strcat(sourceDir, '80MAT_TRANSFORMED/');
if ismember(score_blob, {'h','o','p'})
    det_root = [det_root(1:end-1) '_' score_blob '/'];
end

% set res file
res_file = '%seval_result_%s.mat';
res_file = sprintf(res_file, sourceDir, eval_mode);
if ismember(score_blob, {'h','o','p'})
    res_file = [res_file(1:end-4) '_' score_blob '.mat'];
end

% load annotations
anno = load(anno_file);
bbox = load(bbox_file);

% get gt bbox
switch image_set
    case 'train2015'
        gt_bbox = bbox.bbox_train;
        list_im = anno.list_train;
        anno_im = anno.anno_train;
    case 'test2015'
        gt_bbox = bbox.bbox_test;
        list_im = anno.list_test;
        anno_im = anno.anno_test;
    otherwise
        error('image_set error\n');
end
assert(numel(gt_bbox) == numel(list_im));

% copy variables
list_action = anno.list_action;
num_action = numel(list_action);
num_image = numel(gt_bbox);

% get object list
list_coco_obj = get_list_coco_obj();

% get HOI index intervals for object classes
obj_hoi_int = zeros(numel(list_coco_obj), 2);
for i = 1:numel(list_coco_obj)
    hoi_int = cell_find_string({list_action.nname}', list_coco_obj{i});
    assert(~isempty(hoi_int));
    obj_hoi_int(i, 1) = hoi_int(1);
    obj_hoi_int(i, 2) = hoi_int(end);
end

fprintf('start evaluation\n');
fprintf('setting:     %s\n', eval_mode);
fprintf('exp_name:    %s\n', exp_name);
fprintf('score_blob:  %s\n', score_blob)
fprintf('\n')

if exist(res_file, 'file')
    % load result file
    fprintf('results loaded from %s\n', res_file);
    ld = load(res_file);
    AP = ld.AP;
    REC = ld.REC;
    % print ap for each class
    for i = 1:num_action
        nname = list_action(i).nname;
        aname = [list_action(i).vname_ing '_' list_action(i).nname];
        fprintf('  %03d/%03d %-30s', i, num_action, aname);
        fprintf('  ap: %.4f  rec: %.4f\n', AP(i), REC(i));
    end
else
    % convert gt format
    gt_all = cell(num_action, num_image);
    fprintf('converting gt bbox format ... \n')
    for i = 1:num_image
        assert(strcmp(gt_bbox(i).filename, list_im{i}) == 1)
        for j = 1:numel(gt_bbox(i).hoi)
            if ~gt_bbox(i).hoi(j).invis
                hoi_id = gt_bbox(i).hoi(j).id;
                bbox_h = gt_bbox(i).hoi(j).bboxhuman;
                bbox_o = gt_bbox(i).hoi(j).bboxobject;
                conn = gt_bbox(i).hoi(j).connection;
                boxes = zeros(size(conn, 1), 8);
                for k = 1:size(conn, 1)
                    boxes(k, 1) = bbox_h(conn(k, 1)).x1;
                    boxes(k, 2) = bbox_h(conn(k, 1)).y1;
                    boxes(k, 3) = bbox_h(conn(k, 1)).x2;
                    boxes(k, 4) = bbox_h(conn(k, 1)).y2;
                    boxes(k, 5) = bbox_o(conn(k, 2)).x1;
                    boxes(k, 6) = bbox_o(conn(k, 2)).y1;
                    boxes(k, 7) = bbox_o(conn(k, 2)).x2;
                    boxes(k, 8) = bbox_o(conn(k, 2)).y2;
                end
                gt_all{hoi_id, i} = boxes;
            end
        end
    end
    fprintf('done.\n');

    % load detection
    switch format
        case 'obj'
            % dummy variable
            all_boxes = zeros(num_action, 1);
        case 'all'
            % load detection res (all object mode)
            det_file = [det_root 'detections.mat'];
            ld = load(det_file);
            all_boxes = ld.all_boxes;
    end

    % start parpool
    if ~exist('pool_size','var')
        poolobj = parpool();
    else
        poolobj = parpool(pool_size);
    end
    
    % warning off
    warning('off','MATLAB:mir_warning_maybe_uninitialized_temporary');

    % compute ap for each class    
    AP = zeros(num_action, 1);
    REC = zeros(num_action, 1);
    fprintf('start computing ap ... \n');
    parfor i = 1:num_action
        nname = list_action(i).nname;
        aname = [list_action(i).vname_ing '_' list_action(i).nname];
        fprintf('  %03d/%03d %-30s', i, num_action, aname);
        tic;
        % get detection results
        switch format
            case 'obj'
                % get object id and action id within the object category
                obj_id = cell_find_string(list_coco_obj, nname);
                act_id = i - obj_hoi_int(obj_id, 1) + 1;  %#ok
                assert(numel(obj_id) == 1);
                % load detection res (one object mode)
                det_file = [det_root 'detections_' num2str(obj_id,'%02d') '.mat'];
                ld = load(det_file);
                det = ld.all_boxes(act_id, :);
           case 'all'
                det = all_boxes(i, :);
        end
        % convert detection results
        det_id = zeros(0, 1);
        det_bb = zeros(0, 8);
        det_conf = zeros(0, 1);
        for j = 1:numel(det)
            if ~isempty(det{j})
                num_det = size(det{j}, 1);
                det_id = [det_id; j * ones(num_det, 1)];
                det_bb = [det_bb; det{j}(:, 1:8)];
                det_conf = [det_conf; det{j}(:, 9)];
            end
        end
        % convert zero-based to one-based indices
        det_bb = det_bb + 1;
        % get gt bbox
        assert(numel(det) == numel(gt_bbox));
        gt = gt_all(i, :);
        % adjust det & gt bbox by the evaluation mode
        switch eval_mode
            case 'def'
                % do nothing
            case 'ko'
                nid = cell_find_string({list_action.nname}', nname);  %#ok
                iid = find(any(anno_im(nid, :) == 1, 1));             %#ok
                assert(all(cellfun(@(x)isempty(x),gt(setdiff(1:numel(gt), iid)))) == 1);
                keep = ismember(det_id, iid);
                det_id = det_id(keep);
                det_bb = det_bb(keep, :);
                det_conf = det_conf(keep, :);
        end
        % compute ap
        [rec, prec, ap] = VOCevaldet_bboxpair(det_id, det_bb, det_conf, gt, ...
            min_overlap, aname, false);
        AP(i) = ap;
        if ~isempty(rec)
            REC(i) = rec(end);
        end
        fprintf('  ap: %.4f  rec: %.4f', ap, REC(i));
        fprintf('  time: %.3fs\n', toc);
    end
    fprintf('done.\n');
    
    % warning on
    warning('on','MATLAB:mir_warning_maybe_uninitialized_temporary');

    % delete parpool
    delete(poolobj);
    
    % save AP
    save(res_file, 'AP', 'REC');
end

% get number of instances for each class
num_inst = zeros(num_action, 1);
for i = 1:numel(bbox.bbox_train)
    for j = 1:numel(bbox.bbox_train(i).hoi)
        if ~bbox.bbox_train(i).hoi(j).invis
            hoi_id = bbox.bbox_train(i).hoi(j).id;
            num_inst(hoi_id) = ...
                num_inst(hoi_id) + size(bbox.bbox_train(i).hoi(j).connection,1);
        end
    end
end
s_ind = num_inst < 10;
p_ind = num_inst >= 10;

fprintf('\n');
fprintf('setting:     %s\n', eval_mode);
fprintf('exp_name:    %s\n', exp_name);
fprintf('score_blob:  %s\n', score_blob)
fprintf('\n');
fprintf('  mAP / mRec (full):      %.4f / %.4f\n', mean(AP), mean(REC));
fprintf('\n');
fprintf('  mAP / mRec (rare):      %.4f / %.4f\n', mean(AP(s_ind)), mean(REC(s_ind)));
fprintf('  mAP / mRec (non-rare):  %.4f / %.4f\n', mean(AP(p_ind)), mean(REC(p_ind)));
fprintf('\n');

% output result to .csv file
switch eval_mode
    case 'def'
        fid = fopen(strcat(sourceDir, 'eval_result.txt'),'w');
    case 'ko'
        fid = fopen(strcat(sourceDir, 'eval_result.txt'),'a');
end

fprintf(fid,'\n');
fprintf(fid,'setting:     %s\n', eval_mode);
fprintf(fid,'exp_name:    %s\n', exp_name);
fprintf(fid,'score_blob:  %s\n', score_blob)
fprintf(fid,'\n');
fprintf(fid,'  mAP / mRec (full):      %.4f / %.4f\n', mean(AP), mean(REC));
fprintf(fid,'\n');
fprintf(fid,'  mAP / mRec (rare):      %.4f / %.4f\n', mean(AP(s_ind)), mean(REC(s_ind)));
fprintf(fid,'  mAP / mRec (non-rare):  %.4f / %.4f\n', mean(AP(p_ind)), mean(REC(p_ind)));
fprintf(fid,'\n');
fclose(fid);

