function [imf] = emd(x)                          %program to calculate emd of x
% Empiricial Mode Decomposition (Hilbert-Huang Transform)
% imf = emd(x)
% Func : findpeaks
flag = 0;
j = 1;
k = 1;
x = transpose(x(:));
imf = [];

while ~ismonotonic(x)                                % to check if the signal is not monotonic    
    x1 = x;
    sd = Inf;
    k=1;
    while (sd > 0.1) || ~isimf(x1)
    %while isimf(x1)
      s1 = getspline(x1);                                 %get upper and lower envelope by spline interpolation
      s2 = -getspline(-x1);
      x2 = x1 - (s1+s2)/2;                                  % subtracts the mean of envelope
      
      sd = sum((x1-x2).^2)/sum(x1.^2);                     
      x1 = x2;
      a(j) = k;
      k=k+1;
      if(k==300)                                          % to stop if the decomposition for imf falls in an ever ending loop
          break;
      
      end
  end
   j=j+1;
   imf{end+1} = x1;
   x          = x-x1;
   if(j==20)
       break;
   end
end

imf{end+1} = x;



function u = ismonotonic(x)                             %to check if the signal has decomposed all imfs and has turned monotonic                  
 
u1 = length(findpeaks(x))*length(findpeaks(-x));
if u1 > 0
    u = 0;
else
    u = 1;
end



function u = isimf(x)    %check for stopping criteria of a imf

N  = length(x);
u1 = sum(x(1:N-1).*x(2:N) < 0);
u2 = length(findpeaks(x))+length(findpeaks(-x));
 
if abs(u1-u2) > 1, u = 0;
else,              u = 1;

 end
        



function s = getspline(x)                        %spline interpolation of x
 
  N = length(x);
  p = findpeaks(x);
  s = spline([0 p N+1],[0 x(p) 0],1:N);






