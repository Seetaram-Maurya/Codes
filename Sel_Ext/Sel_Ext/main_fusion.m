%% CWRU
CWRU_hlevel=cell2mat(hlevel_data)';
CWRU_FusedFeatures= [[CWRU_FeaturesNormal;CWRU_FeaturesInner;CWRU_FeaturesBall;CWRU_FeaturesOuter],CWRU_hlevel];
%% AIR
Air_FusedFeatures= [[healthy_features;LIV_features;LOV_features;Piston_features],HLFeaturesHe_LIV_LOV_piston];

%% DB