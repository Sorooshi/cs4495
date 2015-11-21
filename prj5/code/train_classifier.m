function [w,b] = train_classifier(pos_feats, neg_feats)
pos_feats = pos_feats';
neg_feats = neg_feats';
[~, pos_num] = size(pos_feats);
[~, neg_num] = size(neg_feats);
labels = ones(1, (pos_num+neg_num));
labels(pos_num+1:end) = labels(pos_num+1:end)*-1;
[w,b] = vl_svmtrain([pos_feats, neg_feats], labels, 0.0001);

