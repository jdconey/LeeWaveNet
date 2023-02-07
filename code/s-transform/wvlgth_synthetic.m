for threshold = [0,0.125,0.25,0.5,1]
for c2 = [4,0.25]
for k = 0:49
    disp(k)
    data =load("C:/Users/mm16jdc/Documents/ukv_data/data5_testn/S_transform3n/"+threshold+"/mat/"+k+".mat").vv
    
    scale_t = 80
    ps = ones(1,2)*2
    
    
    c = ones(1,2)*c2
    scales = cell(1,2)
    disp(k)
    for i = 1:2
        if i < 2
            scales{i} = -scale_t:scale_t;
        else
            scales{i} = 1:scale_t;
        end
    end
    
    
    ssd = nph_ndst(data,scales,ps,c)
    
    wavelength = (ssd.F1^2+ssd.F2^2)^(-0.5)
    
    angle = atan(ssd.F1/ssd.F2)
    
    F1 = ssd.F1
    F2 = ssd.F2
    save("C:/Users/mm16jdc/Documents/ukv_data/data5_testn/S_transform3n/"+threshold+"/ssd/F1_"+k+"_"+scale_t+"_"+c2+".mat","F1")
    save("C:/Users/mm16jdc/Documents/ukv_data/data5_testn/S_transform3n/"+threshold+"/ssd/F2_"+k+"_"+scale_t+"_"+c2+".mat","F2")
    imshow(ssd.R)
end
end
end