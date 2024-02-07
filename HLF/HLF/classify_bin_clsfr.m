function [res_clsfy] = classify_bin_clsfr(test_vec, bin_clsfr_struct)

[res_clsfy, accuracy] = svmpredict(ones(size(test_vec,1),1), test_vec , bin_clsfr_struct); 

end 