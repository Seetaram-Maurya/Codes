Start=1;
End=150;
for i=1:136
    innerracedata1_5(i,:)=Untitled5(Start:End,5);
    Start=End+1;
    End=(i+1)*150;
end


Start=1;
End=150;
for i=1:136
    rollerdata2_1(i,:)=Untitled1(Start:End,1);
    Start=End+1;
    End=(i+1)*150;
end

IMSInnerFailure=[innerracedata1_1;innerracedata1_2;innerracedata1_3;innerracedata1_4;innerracedata1_5];
IMSOuterFailure=[rollerdata2_1;rollerdata2_2;rollerdata2_3;rollerdata2_4;rollerdata2_5];