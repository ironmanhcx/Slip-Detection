%% Prepare the dataset for Slip Detection with the tactile signals from five-fingered hand.
clear
%clc
close all

directory_path = '~/Documents/MATLAB/dataset/slip_detection/20260129/data/';
% txtfilename = {'shape_slide_S1_30trial.txt', 'shape_slide_S2_30trial.txt', ...
%     'shape_slide_S3_30trial.txt', 'shape_slide_S4_30trial.txt', ...
%     'shape_slide_S5_30trial.txt', 'shape_slide_S6_30trial.txt'};

% 10 types of materials: M1: ABS plastic; M2: Glossy Aluminum foil; M3:
% figured Aluminum foil (fibre-glass reinforced); M4: Brown paper; M5:
% Acrylic sheet; M6: PVC Tape; M7: iron; M8: duct tape; M9: Copper; M10: wood. 

Number_material = 10;
Number_trial = 30;
Number_SE = 24;
Number_Speed = 3;
tactiledata_adc = cell(Number_material, Number_trial, Number_SE);
fs = 2000;

% Data files name generation.
inputfilename = {'abs', 'lb', 'bxlb', 'ypz', 'ykl', 'dg', 'Iron', 'bj', 'Cu', 'wood'};
txtfilename = cell(Number_material, Number_trial);
for i = 1:Number_material
    for j = 1:Number_trial
        txtfilename{i, j} = [inputfilename{i}, '-slip', num2str(j), '.txt'];
    end
end

% Duration_sliding = 10;
% Duration_pause1 = 3;
% Duration_pause2 = 2;
% CenterPoint = (Duration_pause1+Duration_sliding+Duration_pause2)/2;
% DurationPoint = (Duration_pause1+Duration_sliding+Duration_pause2)*fs;

%% Related parameters.
para.nummaterial = Number_material;
para.numtrial = Number_trial;
para.numSE = Number_SE;
para.numspeed = Number_Speed;
para.speed = [0.02, 0.04, 0.06];
para.slipDist = 0.1;
para.staticT = 2;
para.fs = fs;
% para.durationp1sp2 = [Duration_pause1, Duration_sliding, Duration_pause2];

% % % %% Step 1: filter the signal with notching filter, i.e., remove components of 50 Hz, 100 Hz, 150 Hz, ...
% % % Q = 30;
% % % harmonics = 150:50:450; % folds of 50Hz


%% Filter the signal with Savitzky-Golay filter.
order_PVDF = 1;
order_SG = 1;
framelen_pvdf = 11;
framelen_SG = 51;
data_filtered = cell(Number_material, Number_trial, Number_SE);

index_PVDF = 15:24;
index_SG = 1:14;

randsetting = rng;
% rng(randsetting);
linecolor = rand(Number_SE, 3);
% truth_slip = [2, 9, 13.5];
truth_sliplabelled = readtable('/home/longhuiqin/Documents/MATLAB/Codes/slip_detection/labelled_groundtruth.xlsx');
truth_slip_point = table2array(truth_sliplabelled(:, 2:end));
truth_slip = truth_slip_point/para.fs;
kk=1;
for i = 1:Number_material
    for j = 1:Number_trial
        datasource = readtable(fullfile(directory_path,txtfilename{i, j}));
        %% plot the original sNumber_shapeNumber_shapeignal
        for k = 1:Number_SE
            tactiledata_adc{i, j, k} = table2array(datasource(:,k+1))';
        end
        %% filtering
        for k = index_PVDF
            data_filtered{i, j, k} = sgolayfilt(tactiledata_adc{i, j, k}, order_PVDF, framelen_pvdf);
        end
        for k = index_SG
            data_filtered{i, j, k} = sgolayfilt(tactiledata_adc{i, j, k}, order_SG, framelen_SG);
        end
    end
end

% %% plot the signal
% j = 1; % j: Trial number, ranging within [1, 30]. i: Material index, ranging within [1, 10].
% for i = 1:Number_material
%     figure(kk)
%     for k = index_SG
%         plot((1:numel(data_filtered{i, j, k}))/fs, data_filtered{i, j, k}, 'Color', linecolor(k,:), 'LineStyle','-', 'LineWidth', 1.5)
%         hold on
%     end
%     xline(truth_slip(j,3*(i-1)+(1:3)), '--k', 'LineWidth', 2)
%     xlim([0 (numel(data_filtered{i, j, k})/fs+0.1)])
%     xlabel('Time/s')
%     ylabel('Voltage/V')
%     title(['M', num2str(i), ', Trial ', num2str(j), ', SG'])
%     kk = kk+1;
% 
%     figure(kk)
%     for k = index_PVDF
%         plot((1:numel(data_filtered{i, j, k}))/fs, data_filtered{i, j, k}, 'Color', linecolor(k,:), 'LineStyle','-', 'LineWidth', 1.5)
%         hold on
%     end
%     xline(truth_slip(j,3*(i-1)+(1:3)), '--k', 'LineWidth', 2)
%     xlim([0 (numel(data_filtered{i, j, k})/fs+0.1)])
%     xlabel('Time/s')
%     ylabel('Voltage/V')
%     title(['M', num2str(i), ', Trial ', num2str(j), ', PVDF'])
%     kk = kk+1;
% end

%% Signal partition.
%% the duration for some trials is smaller than expected.
%% The unsuitable trials are removed and the trial amount reduces from 30 to 28.
data_static = cell(Number_material, Number_trial-2, Number_SE, Number_Speed); % data during static period.
%% staticwidth: valid width for static period.
para.staticwidth = 1; %% 1 second is selected.
para.staticduration = round(para.staticwidth*fs);
para.speed = [0.02, 0.04, 0.06];
para.slipDist = 0.1;
para.slidingduration = round(para.slipDist./para.speed.*fs);
data_sliding = cell(Number_material, Number_trial-2, Number_SE, Number_Speed); % data during static period.
para.onsetadance = 0.01; %% duration ahead of the onset time
% para.slidingduration3 = zeros(Number_material, Number_trial); % In some cases, the sliding duration may be less than 1.67s.
truth_slip_pointr = truth_slip_point(1:28,:);
pair2remove = [2,18; 7,24; 8,4; 10,17];
for i = 1:Number_material
    for j = 1:(Number_trial-2)
        temp = pair2remove-[i,j];
        TFlag = ismember([0,0], temp, 'rows');
        if TFlag
            Ntrial = 29;
            truth_slip_pointr(j,3*(i-1)+(1:3)) = truth_slip_point(Ntrial,3*(i-1)+(1:3));
        else
            Ntrial = j;
        end
        time_onset = truth_slip_point(Ntrial,3*(i-1)+(1:3))-round(fs*para.onsetadance);
        time_staticperiod = [(time_onset(1)-para.staticduration), time_onset(1)-1, (time_onset(2)-para.staticduration), time_onset(2)-1, (time_onset(3)-para.staticduration), time_onset(3)-1];
        % if (time_onset(3)+para.slidingduration(3)-1)>numel(data_filtered{i, Ntrial, 1})
        %     para.slidingduration3(i,j) = numel(data_filtered{i, Ntrial, 1})-time_onset(3)+1;
        % else
        %     para.slidingduration3(i,j) = para.slidingduration(3);
        % end

        for k = 1:Number_SE
            data_static{i,j,k,1} = data_filtered{i, Ntrial, k}(time_staticperiod(1):time_staticperiod(2));
            data_static{i,j,k,2} = data_filtered{i, Ntrial, k}(time_staticperiod(3):time_staticperiod(4));
            data_static{i,j,k,3} = data_filtered{i, Ntrial, k}(time_staticperiod(5):time_staticperiod(6)); 
            data_sliding{i,j,k,1} = data_filtered{i, Ntrial, k}(time_onset(1):(time_onset(1)+para.slidingduration(1)-1));
            data_sliding{i,j,k,2} = data_filtered{i, Ntrial, k}(time_onset(2):(time_onset(2)+para.slidingduration(2)-1));
            data_sliding{i,j,k,3} = data_filtered{i, Ntrial, k}(time_onset(3):(time_onset(3)+para.slidingduration(3)-1));
        end
    end
end

para.index_SG = index_SG;
para.index_PVDF = index_PVDF;
truth_slipr = truth_slip_pointr/para.fs;
para.truth_slip_point = truth_slip_pointr;
para.truth_slip = truth_slipr;
para.numtrial = Number_trial-2;

save('dataset_M10Trial30Speed3v1.mat', 'tactiledata_adc', 'data_filtered', 'para', 'randsetting', ...
    'data_static', 'data_sliding');