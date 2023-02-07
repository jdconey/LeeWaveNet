for k = 1:28 
    if k < 10
        k="0"+k
    end
    data =load("s_transform\data\vv_no_mask"+k+".mat").vv
    
    c2=1

    c= ones(1,2)*c2
    
    ps = ones(1,2)*2
    
    scales = cell(1,2)
    for i = 1:2
        if i < 2
            scales{i} = -80:80;
        else
            scales{i} = 1:80;
        end
    end
    
    
    ssd = nph_ndst(data,scales,ps,c)
    
    wavelength = (ssd.F1^2+ssd.F2^2)^(-0.5)
    
    angle = atan(ssd.F1/ssd.F2)
    
    F1 = ssd.F1
    F2 = ssd.F2
    imshow(ssd.R)
    save("C:/Users/mm16jdc/Documents/MATLAB/80/F1_"+k+"_"+c2+".mat","F1")
    save("C:/Users/mm16jdc/Documents/MATLAB/80/F2_"+k+"_"+c2+".mat","F2")
end