%% Prepare the dataset for Slip Detection with the tactile signals from five-fingered hand.
clear
%clc
close all

load('dataset_M10Trial30Speed3v1.mat', 'tactiledata_adc', 'data_filtered', 'para', 'truth_slip', 'randsetting', ...
    'data_static', 'data_sliding');

% 10 types of materials: M1: ABS plastic; M2: Glossy Aluminum foil; M3:
% figured Aluminum foil (fibre-glass reinforced); M4: Brown paper; M5:
% Acrylic sheet; M6: PVC Tape; M7: iron; M8: duct tape; M9: Copper; M10: wood. 

%% signal in static period
data_static_total = cell(para.nummaterial, para.numtrial);
%% signal in sliding period
data_sliding_total = cell(para.nummaterial, para.numtrial, para.numspeed);

for i = 1:para.nummaterial
    for j = 1:para.numtrial
        temp_static = [];
        temp_sliding_v1 = [];
        temp_sliding_v2 = [];
        temp_sliding_v3 = [];
        for k = 1: para.numSE
            temp_static = [temp_static; [data_static{i,j,k,1}, data_static{i,j,k,2}, data_static{i,j,k,3}]];
            temp_sliding_v1 = [temp_sliding_v1; data_sliding{i,j,k,1}];
            temp_sliding_v2 = [temp_sliding_v2; data_sliding{i,j,k,2}];
            temp_sliding_v3 = [temp_sliding_v3; data_sliding{i,j,k,3}];
        end
        data_static_total{i,j} = temp_static;
        data_sliding_total{i,j,1} = temp_sliding_v1;
        data_sliding_total{i,j,2} = temp_sliding_v2;
        data_sliding_total{i,j,3} = temp_sliding_v3;
    end
end

%% application of binning technique.
binwidthT = 0.05;
binwidth = round(binwidthT*para.fs);
Nbin_static1trial = floor((para.staticduration*para.numspeed)/binwidth); % quantity of bins within 1 trial.
Nbin_static = Nbin_static1trial*para.numtrial;

signal_static_binned = cell(para.nummaterial, Nbin_static);

Nbin_sliding1 = floor((para.slidingduration(1)*para.numtrial)/binwidth);
signal_sliding1_binned = cell(para.nummaterial, Nbin_sliding1);  % speed 1.
Nbin_sliding2 = floor((para.slidingduration(2)*para.numtrial)/binwidth);
signal_sliding2_binned = cell(para.nummaterial, Nbin_sliding2);  % speed 2.
Nbin_sliding3 = floor((para.slidingduration(3)*para.numtrial)/binwidth);
signal_sliding3_binned = cell(para.nummaterial, Nbin_sliding3);  % speed 3.

% static data
for i = 1:para.nummaterial
    kk = 1;
    for j = 1:para.numtrial
        for n = 1:Nbin_static1trial
            signal_static_binned{i,kk} = data_static_total{i,j}(((n-1)*binwidth+1):(n*binwidth));
            kk = kk+1;
        end
    end
end

% sliding data at speed v1.
for i = 1:para.nummaterial
    kk = 1;
    for j = 1:para.numtrial
        for n = 1:Nbin_sliding1
            signal_sliding1_binned{i,kk} =  data_sliding_total{i,j,1}(((n-1)*binwidth+1):(n*binwidth));
            kk = kk+1;
        end
    end
end
% sliding data at speed v2.
for i = 1:para.nummaterial
    kk = 1;
    for j = 1:para.numtrial
        for n = 1:Nbin_sliding2
            signal_sliding2_binned{i,kk} =  data_sliding_total{i,j,2}(((n-1)*binwidth+1):(n*binwidth));
            kk = kk+1;
        end
    end
end
% sliding data at speed v3.
for i = 1:para.nummaterial
    kk = 1;
    for j = 1:para.numtrial
        for n = 1:Nbin_sliding3
            signal_sliding3_binned{i,kk} =  data_sliding_total{i,j,3}(((n-1)*binwidth+1):(n*binwidth));
            kk = kk+1;
        end
    end
end







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
truth_slip = [2, 9, 13.5];
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

%% plot the signal
j = 1; % j: Trial number, ranging within [1, 30]. i: Material index, ranging within [1, 10].
for i = 1:Number_material
    figure(kk)
    for k = index_SG
        plot((1:numel(data_filtered{i, j, k}))/fs, data_filtered{i, j, k}, 'Color', linecolor(k,:), 'LineStyle','-', 'LineWidth', 1.5)
        hold on
    end
    xline(truth_slip, '--k', 'LineWidth', 2)
    xlim([0 (numel(data_filtered{i, j, k})/fs+0.1)])
    xlabel('Time/s')
    ylabel('Voltage/V')
    title(['M', num2str(i), ', Trial ', num2str(j), ', SG'])
    kk = kk+1;

    figure(kk)
    for k = index_PVDF
        plot((1:numel(data_filtered{i, j, k}))/fs, data_filtered{i, j, k}, 'Color', linecolor(k,:), 'LineStyle','-', 'LineWidth', 1.5)
        hold on
    end
    xline(truth_slip, '--k', 'LineWidth', 2)
    xlim([0 (numel(data_filtered{i, j, k})/fs+0.1)])
    xlabel('Time/s')
    ylabel('Voltage/V')
    title(['M', num2str(i), ', Trial ', num2str(j), ', PVDF'])
    kk = kk+1;
end

save('dataset_M10Trial30Speed3v1.mat', 'tactiledata_adc', 'data_filtered', 'para', 'truth_slip', 'randsetting');