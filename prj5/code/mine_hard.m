function [w,b] = mine_hard(pos_feats, neg_feats, w, b, non_face_scn_path, feature_params)
pos_feats = pos_feats';
neg_feats = neg_feats';
[~, pos_num] = size(pos_feats);
[~, neg_num] = size(neg_feats);
labels = ones(1, (pos_num+neg_num));
labels(pos_num+1:end) = labels(pos_num+1:end)*-1;
all_feats = [pos_feats, neg_feats];

new_neg_feats = negtive_mining(non_face_scn_path, w, b, feature_params)';
num_new = size(new_neg_feats, 2);
labels = [labels, ones(1, num_new)*-1];
all_feats = [all_feats, new_neg_feats];
[w,b] = vl_svmtrain(all_feats, labels, 0.0001);
