function [new_neg_feats] = negtive_mining(non_face_scn_path, w, b, feature_params)
neg_scenes = dir( fullfile( non_face_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.

cell_size = feature_params.hog_cell_size;
range = feature_params.template_size/cell_size;
new_neg_feats = zeros(0, range^2*31);
for i = 1:length(neg_scenes)
      
    img = imread( fullfile( non_face_scn_path, neg_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    
    cur_confidences = zeros(0,1);
    cur_bboxes = zeros(0,4);
    cur_image_ids = cell(0,1);
    scale_min = 36 / min(size(img));
    scale_max = 1;
    scale_range = scale_min : 0.05 : scale_max;
    for d = scale_range
        scale_img = imresize(img, d);
        hog_space = vl_hog(scale_img, feature_params.hog_cell_size);
        h = floor(size(scale_img, 1) / feature_params.hog_cell_size);
        width = floor(size(scale_img, 2) / feature_params.hog_cell_size);
         
        window_hog = zeros((h-range+1)*(width-range+1), range^2*31);
        for k = 1 : h-range+1
            for j = 1 : width-range+1
                patch = hog_space(k:k+range-1, j:j+range-1, :);
                patch = reshape(patch, 1, range^2*31);
                window_hog((k-1)*(width-range+1)+j, :) = patch;
            end
        end
        
        scores = window_hog * w + b;
        indices = find(scores > -0.5);
        scale_confidences = scores(indices);
        
        row_num = floor(indices / (width - range + 1));
        col_num = mod(indices, width - range + 1);
        row_num(find(col_num == 0)) = row_num(find(col_num == 0)) - 1;
        col_num(find(col_num == 0)) = width - range + 1;
        col_num = col_num - 1;
        scale_bboxes = [cell_size*col_num+1, cell_size*row_num+1, cell_size*(col_num+range), cell_size*(row_num+range)]/d;
        scale_image_ids = repmat({neg_scenes(i).name},  size(indices,1), 1);
        
        cur_bboxes = [cur_bboxes; scale_bboxes];
        cur_confidences = [cur_confidences; scale_confidences];
        cur_image_ids = [cur_image_ids; scale_image_ids];
    end
    
    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));

    cur_bboxes      = cur_bboxes(     is_maximum,:);

    cur_bboxes = ceil(cur_bboxes);
    in_bboxes = cur_bboxes(find(cur_bboxes(:,1) > 0 & ...
        cur_bboxes(:,2) > 0 & cur_bboxes(:,3) <= size(img, 2) & cur_bboxes(:,4) <= size(img, 1)), :);
    neg_feats = zeros(size(in_bboxes, 1), range^2*31);
    for m = 1 : size(in_bboxes,1)
        new_patch = imresize(img(in_bboxes(m, 2):in_bboxes(m, 4), in_bboxes(m, 1):in_bboxes(m, 3)), ...
            [feature_params.template_size, feature_params.template_size]);
        neg_feats(m, :) = reshape(vl_hog(new_patch, cell_size), 1, range^2*31);
    end
    new_neg_feats = [new_neg_feats; neg_feats];
end

