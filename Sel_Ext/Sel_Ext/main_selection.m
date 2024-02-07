num_fea=5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Air_FusedHePistonFeaturesLab=[[Air_FusedFeatures(1:225,:);Air_FusedFeatures(676:900,:)],append];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Selecting Best features
% A) Using PCA
sel_fea_pca=find_pca(Air_FusedHePistonFeaturesLab(:,1:60),num_fea);
% B) Using MIFS
sel_fea_mifs=find_MI_ind(Air_FusedHePistonFeaturesLab,1,0.5,num_fea);
% F) Using BDA
sel_fea_bd=find_BD_ind(Air_FusedHePistonFeaturesLab(:,1:60),num_fea);
%%%%%%%%%%%%%%%%%%%%%%%


