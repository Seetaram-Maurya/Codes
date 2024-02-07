



function [processed_data]= fun_preprocessing(rawdata, sampling_freq,coeff_file)
          clipped_data=clipping(rawdata,10,sampling_freq);
          fanfilter_data=fun_fan_filter(clipped_data,coeff_file);
          filtered_data=butterfilter(fanfilter_data,sampling_freq,11000);
          smoothed_data=smoothing(filtered_data);
          %processed_data=minmax_normalization(smoothed_data,sampling_freq);
          processed_data=smoothed_data;
end

function [rawdata]=wavread(path)
    data=wavread(path);
    rwdata=lov(:,1);
    rawdata=rwdata;
end

function [rawdata]=datread(path)
          data=load(path);
          data1=data(1,:);
          rawdata=data1';
end


% Clipping             
function [clipped_data]=clipping(data,num_of_parts,sampling_freq)
%             num_datapoints=length(data)/num_of_parts;
            num_datapoints=22050;
            for i=1:(num_of_parts-1)
                x= (i-1)*num_datapoints;
                y= i*sampling_freq - (i-1)*num_datapoints;
                a=data(x+1:y);
                all_std(i)=std(a);
            end
            [a b]=min(all_std);
            clipped_data=data(((b-1)*num_datapoints+1):((b)*sampling_freq - ((b-1)*num_datapoints)));
            clear s a i rawdata x y data; 
end

%Filtering using fan filter
%  function [filtered_data]=fun_fan_acoustic_filter(data)
%                                                         %d is the actual signal
% data=data(1,:); 
% fd=fft(data);
% db_d=20*log10(abs(fd));                                                     %amplitude in decibles
% fst=(200*pi)/22050;                                                         %stopband frequency(200) normalized(0-1)
% fp=(300*pi)/22050;                                                          %pass frequency(300) normalized(0-1)
% high=fdesign.highpass(fst,fp,30,1);                                         %parameters:"f_stopband, f_passband, stopband_attenuation, passband_ripple"
% Hdh=design(high);
% filtered_data=filter(Hdh,data);                                                            %applying filter on e
% fid=fopen('filtered.dat','w');
% fprintf(fid,'%f  ' ,filtered_data);
% fclose(fid);
% close all;
% end
% 

% Filtering using butter worth filter
function [y]=fun_fan_filter(x,coeff)
% fid=fopen(coeff_file,'r');
% coeff=fscanf(fid,'%f\t');  
% %coefficient of high pass filter(fst=350,fp=450,Ap=1;As=60)
% fclose(fid);clear fid;
coeff=coeff';
y=filter(coeff,[1],x);  
end
function [filtered_data]=butterfilter(data,sampling_freq,filter_freq)
            
           Ts = 1/sampling_freq; %sampling time interval   
           [B, A]= butter(18, (filter_freq*2 / (sampling_freq)), 'low');
           filtered_data = filter(B,A,data); 
           clear A B Ts sampling_freq filter_freq data; 
end
% Smoothing using moving average method
function [smooth_data]=smoothing(filter_data)
            smooth_data = zeros(1,44100);
            smooth_data(1)=(filter_data(1)+filter_data(2)+filter_data(3))/3;
            smooth_data(2)=(filter_data(1)+filter_data(2)+filter_data(3)+filter_data(4))/4;
            for j=3:44098
                smooth_data(j)=(filter_data(j-2)+filter_data(j-1)+filter_data(j)+filter_data(j+1)+filter_data(j+2))/5;
            end
            smooth_data(44099) = (filter_data(44097)+filter_data(44098)+filter_data(44099)+filter_data(44100))/4;
            smooth_data(44100) = (filter_data(44098)+filter_data(44099)+filter_data(44100))/3;
            clear filter_data j;
end

% Normalization using min-max.
function [normalized_data]=minmax_normalization(data,sample_frequency)
       min_data = min(data);
       max_data = max(data);
       normalized_data = (data - repmat(min_data,1,sample_frequency))./repmat(max_data-min_data,1,length(data));
       clear sample_frequency data;
end