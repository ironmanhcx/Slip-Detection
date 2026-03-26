% % % Detection of onset time for slip detection.
% % % 10 materials.

clc;
clear
close all;

%% dataset preparation.
directory_path = 'D:/Scholar_SRTP/slip_detection/slip_detection';
load([directory_path, '/dataset_M10Trial30Speed3v1.mat'], 'data_filtered');
load([directory_path, '/dataset_binnedSpeed3v1.mat'], 'para');
load([directory_path, '/slip_classificationELMmodel.mat'], 'Num_feature',  'idxfeature', 'Featureinforsorted');

%% load original data and remove unsuitable trials. Number of trials is 28;
truth_slip = para.truth_slip_point/para.fs;

featureconvertFLAG = 0; %% save time of feature extraction.
if featureconvertFLAG==1

    pair2remove = [2,18; 7,24; 8,4; 10,17];
    signal_28trials = cell(para.nummaterial, para.numtrial);
    Nbin_28trial = zeros(para.nummaterial, para.numtrial);
    num_feature_total = numel(Featureinforsorted.fname);
    for i = 1:para.nummaterial
        for j = 1:para.numtrial
            temp = pair2remove-[i,j];
            TFlag = ismember([0,0], temp, 'rows');
            if TFlag
                Ntrial = 29;
            else
                Ntrial = j;
            end
            for k = 1:para.numSE
                signal_28trials{i,j}(k, :) = data_filtered{i, Ntrial, k};
            end
            Nbin_28trial(i,j) = floor(length(signal_28trials{i,j})/para.binwidth);
        end
    end

    signal_binned = cell(para.nummaterial, para.numtrial, max(max(Nbin_28trial)));
    featurepoolpred = cell(para.nummaterial, para.numtrial);
    SelectFeaturePred = cell(para.nummaterial, para.numtrial);
    for i = 1:para.nummaterial
        for j = 1:para.numtrial
            signal_current = signal_28trials{i,j};
            featurepoolpredtemp = zeros(Nbin_28trial(i,j), num_feature_total);
            SelectFeaturePredtemp = zeros(Nbin_28trial(i,j), Num_feature);
            for n = 1:Nbin_28trial(i,j)
                signal_binned{i,j,n} = signal_current(:, ((n-1)*para.binwidth+1):(n*para.binwidth));
                featureSG = [];
                for k = para.index_SG
                    [featureSGtemp, info] = extract_features_1signal(signal_binned{i,j,n}(k,:), 'SG', para.fs, 0, 'T', 'rbio2.2');
                    featureSG = [featureSG, featureSGtemp];
                end
                featurePVDF = [];
                for k = para.index_PVDF
                    [featurePVDFtemp, info] = extract_features_1signal(signal_binned{i,j,n}(k,:), 'PVDF', para.fs, 4, 'TF', 'rbio2.2');
                    featurePVDF = [featurePVDF, featurePVDFtemp];
                end
                featurepoolpredtemp(n,:) = [featureSG, featurePVDF];
                SelectFeaturePredtemp(n,:) = featurepoolpredtemp(n,idxfeature(1:Num_feature));
            end
            featurepoolpred{i,j} = featurepoolpredtemp;
            SelectFeaturePred{i,j} = SelectFeaturePredtemp;
        end
    end

    save('SelectedFeaturepool10MPred.mat', 'para', 'featurepoolpred', 'SelectFeaturePred','Num_feature', 'signal_28trials', 'Nbin_28trial');
else
    load('SelectedFeaturepool10MPred.mat', 'para', 'featurepoolpred', 'SelectFeaturePred', 'Num_feature', 'signal_28trials', 'Nbin_28trial');
end

%% Feature extraction.
load([directory_path, '/slip_classificationELMmodel.mat'], 'TrainedMdl', 'inputps', 'Num_feature');
% para.onsetadance = 0.01; %% duration ahead of the onset time
% para.staticwidth = 1; %% 1 second is selected.
% para.staticduration = round(para.staticwidth*para.fs);
testFLAG = 0;
if testFLAG==1
    Class_output = cell(para.nummaterial,para.numtrial);
    PredTime = zeros(para.nummaterial,para.numtrial);
    for i = 1:para.nummaterial
        for j = 1:para.numtrial
            Num_bins = Nbin_28trial(i,j);
            output_pred = zeros(1,Num_bins);
            input1_test = SelectFeaturePred{i,j};
            input_test=input1_test';
            %% Normalization.
            inputn_test=mapminmax('apply',input_test,inputps);
            testdata_input=inputn_test';
            testdatafile = zeros(Num_bins, Num_feature+1);
            testdatafile(:,1)=zeros(Num_bins, 1);
            testdatafile(:,2:(size(testdata_input,2)+1))=testdata_input;
            [TestY, ~, TestingTime] = elm_kernel_test(TrainedMdl, testdatafile);
             [~, label_index_actual]=max(TestY, [], 1);
            Class_output{i,j} = label_index_actual-1;
            PredTime(i,j) = TestingTime;
        end
    end

    save('PredTime10M,mat', 'Class_output', 'PredTime');
else
    load('PredTime10M,mat', 'Class_output', 'PredTime');
end

%% Plot the Result.
close all
% ---- Font settings (increase tick + label sizes, and use Times New Roman) ----
fontName = 'Times New Roman';
tickFontSize  = 32;   % tick labels (numbers) on both axes
labelFontSize = 34;   % x/y axis labels
titleFontSize = 26;   % figure title
legendFontSize = 28;  % legend text
% -------------------------------------------------------------------------
% ---- Plot style settings (editable) ------------------------------------
% Bar colors (no slip / slip): use RGB triplets in [0,1] or [0,255]/255
noSlipRGB   = [204,225,255]/255;   % <-- no slip block color
noSlipAlpha = 0.75;                 % 0 (transparent) ~ 1 (opaque)

slipRGB     = [255, 215, 0]/255;   % <-- slip block color
slipAlpha   = 0.28;                 % 0 (transparent) ~ 1 (opaque)

% Signal line style
pvdfLineWidth = 2.2;    % signal line thickness
pvdfDarken    = 0.4;   % 0=no change, 1=black (deeper color)
useDeterministicColors = true;  % true: stable palette; false: random

% Generate (and optionally darken) PVDF line colors
if useDeterministicColors
    linecolor = lines(para.numSE);
else
    linecolor = rand(para.numSE, 3);
end
linecolor = max(0, min(1, linecolor * (1 - pvdfDarken)));
% -------------------------------------------------------------------------
kk = 1;
for i = 1:para.nummaterial
    % for j = 1:para.numtrial
    for j = 1
        figure(kk)
        yyaxis left
        for k = para.index_PVDF
            plot((1:numel(signal_28trials{i, j}(k,:)))/para.fs, signal_28trials{i, j}(k,:), 'Color', linecolor(k,:), 'LineStyle', '-', 'Marker', "none", 'LineWidth', pvdfLineWidth)
            hold on
        end
        xline(para.truth_slip(j,3*(i-1)+(1:3)), '--k', 'LineWidth', 2)
        xlim([0 (numel(signal_28trials{i, j}(k,:))/para.fs+0.1)])
        yticks([-5, 0, 5])
        xlabel('Time/s','FontName',fontName,'FontSize',labelFontSize)
        ylabel('Voltage/V','FontName',fontName,'FontSize',labelFontSize,'Color',[0 0 0])
        title(['M', num2str(i), '-Trial', num2str(j), ''], 'FontName',fontName,'FontSize',titleFontSize)

        x_bin = ((1:Nbin_28trial(i,j))-1+0.5)*para.binwidth/para.fs;
        yyaxis right
        bfig = bar(x_bin, Class_output{i,j}, 1, 'FaceColor', slipRGB, 'EdgeColor', 'none');
        hold on
        bfig2 = bar(x_bin, 1-Class_output{i,j}, 1, 'FaceColor', noSlipRGB, 'EdgeColor', 'none');
        yticks([0, 1])
        % hold off
        % Set bar transparency (prefer FaceAlpha; fall back to alpha for older versions)
        try
            bfig.FaceAlpha = slipAlpha;
        catch
            alpha(bfig, slipAlpha);
        end
        try
            bfig2.FaceAlpha = noSlipAlpha;
        catch
            alpha(bfig2, noSlipAlpha);
        end
        set(gca, 'FontName', fontName, 'FontSize', tickFontSize)
        lgd = legend([bfig2, bfig], {'non-slip', 'slip'});
        set(lgd, 'FontName', fontName, 'FontSize', legendFontSize)
        % ---- Force both y-axes (ticks + axis line) to black (avoid default blue with yyaxis) ----
        ax = gca;
        yyaxis left
        ax.YColor = [0 0 0];
        yyaxis right
        ax.YColor = [0 0 0];
        kk = kk+1;
    end
end


