function updatefigure(fhandle,L_train, L_val, val_flag, numEpochs)
 
    % create legend
    if val_flag == 1
        leg_str = {'Training','Validation'};
    else
        leg_str = {'Training'};
    end
    
    y = L_train; 

    figure(fhandle);
    cla
    set(gca,'Xlim',[0,numEpochs])
    xlabel('Number of epochs');
    ylabel('Loss');

    hold on
    x = 1:length(L_train);
    p1 = plot(x,y);
     
    if val_flag == 1
        
        y_val = L_val;
        p2 = plot(x, y_val);
        legend([p1,p2], leg_str,'Location','NorthEast');
        
    else
        
        legend(p1, leg_str,'Location','NorthEast');

    end
    
    hold off
    drawnow;
