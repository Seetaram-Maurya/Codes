function feature_ind = find_MI_ind(data,Flag,B, num_fea)

%% parameter adjustment
nbin = 15; % number of bin for frequency calcuation
fvdim = size(data,2)-1;                                                   % Initial Feature vecotr size
[MI_ff MI_fc H_FV] = func_mutualinfo(data,nbin);
K = fvdim; % |S| = k ;
S = zeros(K,1);
F = 1:fvdim;
[a S(1)]= max(MI_fc);
F = F(find(F~=S(1)));

for k = 2 : K
    clear mid
    for i =1 : size(F,2)    
        summ = 0;
        if Flag == 1  % MIFS
            for j = 1 : k-1            
                summ = summ + MI_ff(F(i),S(j));            
            end
            mid(i) = MI_fc(F(i)) - summ*B;
        elseif Flag ==2 % mRMR            
            for j = 1 : k-1            
                summ = summ + MI_ff(F(i),S(j));            
            end
            mid(i) = MI_fc(F(i))- summ/(k-1);
        elseif Flag  == 3 % NMIFS
            for j = 1 : k-1            
                N_ff = min(H_FV(j),H_FV(i));
                summ = summ + MI_ff(F(i),S(j))/N_ff;            
            end
            mid(i) = MI_fc(F(i))- summ/(k-1);
        elseif Flag == 4 % MIFS-U
            for j = 1 : k-1
                summ = summ + MI_ff(F(i),S(j))*MI_fc(S(j))/H_FV(S(j));
            end
            mid(i) = MI_fc(F(i))- summ*B;
        else
            'error'
            break;
        end    
    end
    [a b] = max(mid);
    S(k) = F(b);
    F = F(find(F~=S(k)));
    
end

feature_ind = S(1:num_fea);
