% slipdetection_fs_v1_test5_fastbuild_withlabel_v6_fix.m
% K 固定（不再搜索 K），只搜索 cap + C（可选阈值 thr），并修复“数组大小不兼容”问题
clc; clear; close all;
rng(0,'twister');

%% ========= 你只需要改这里 =========
directory_path = 'D:\Scholar_SRTP\slip_detection\slip_detection';  % <-- 改成你自己的路径
addpath(directory_path); addpath(genpath(directory_path));

fullMatPath = fullfile(directory_path, 'SlipBinnedFeature.mat');
leanMatPath = fullfile(directory_path, 'SlipBinnedFeature_6M_v73.mat');

Num_feature = 120;                 % <<< 按你要求：K 不动
trainRatio  = 0.80;
valRatio    = 0.15;

trainCapCandidates = [21000 20000 18000 14000 12000 10000 8000 6000];  % 尽量大（能跑就更准）
C_candidates = [0.05 0.08 0.1 0.13 0.17 0.2 0.25 0.3 1 3];                        % 只调 C

Elm_Type    = 1;
Kernel_type = 'poly_kernel';
Kernel_para = [0.5, 2];

useRankSubset   = true;
rankSubsetTotal = 20000;

keepFeaturepoolAsSingle = true;
testBlockSize = 400;

enableThresholdTuning = true;   % 阈值后处理（在验证集上选阈值）
nThrGrid = 81;                  % 阈值网格数量

%% ========= 0) 载入数据（lean mat 不存在就从 full mat 复制） =========
if exist(leanMatPath,'file') ~= 2
    fprintf('[WARN] lean MAT not found. Copying from full MAT using matfile...\n');
    if exist(fullMatPath,'file') ~= 2
        error('Cannot find %s or %s. Please check paths.', leanMatPath, fullMatPath);
    end
    mf = matfile(fullMatPath);
    info = whos(mf); names = string({info.name});
    need = ["featurepool6M","output_srclass6M"];
    if ~all(ismember(need,names))
        error('Full MAT missing featurepool6M/output_srclass6M.');
    end
    featurepool6M    = mf.featurepool6M;
    output_srclass6M = mf.output_srclass6M;
    save(leanMatPath,'featurepool6M','output_srclass6M','-v7.3');
    clear featurepool6M output_srclass6M;
end

S = load(leanMatPath, 'featurepool6M','output_srclass6M');
featurepool6M = S.featurepool6M;
y_all         = double(S.output_srclass6M(:));  % 0/1
clear S;

fprintf('[INFO] featurepool6M: %dx%d (%s)\n', size(featurepool6M,1), size(featurepool6M,2), class(featurepool6M));
if keepFeaturepoolAsSingle && ~isa(featurepool6M,'single')
    featurepool6M = single(featurepool6M);
    fprintf('[INFO] Cast featurepool6M to single.\n');
end

%% ========= 1) rankfeatures（固定取前 Num_feature 个） =========
N = size(featurepool6M,1);
if useRankSubset && ~isempty(rankSubsetTotal) && rankSubsetTotal < N
    idx0 = find(y_all==0); idx1 = find(y_all==1);
    nEach = floor(rankSubsetTotal/2);
    nEach = min([nEach, numel(idx0), numel(idx1)]);
    subIdx = [idx0(randperm(numel(idx0), nEach)); idx1(randperm(numel(idx1), nEach))];
    subIdx = subIdx(randperm(numel(subIdx)));
    Xrank = double(featurepool6M(subIdx,:));
    yrank = y_all(subIdx);
    fprintf('[INFO] rankfeatures subset: %d (balanced)\n', numel(subIdx));
else
    Xrank = double(featurepool6M);
    yrank = y_all;
    fprintf('[INFO] rankfeatures FULL: %d\n', N);
end

Xrank_norm = mapminmax(Xrank', 0, 1)';   % 归一化后做排序
[idxfeature, ~] = rankfeatures(Xrank_norm', yrank');
clear Xrank Xrank_norm yrank;

X_all = featurepool6M(:, idxfeature(1:Num_feature)); % <<< K 固定

%% ========= 2) 分层 train/test =========
idx0 = find(y_all==0); idx1 = find(y_all==1);
r0 = randperm(numel(idx0)); r1 = randperm(numel(idx1));
n0_tr = round(numel(idx0)*trainRatio);
n1_tr = round(numel(idx1)*trainRatio);

tr0 = idx0(r0(1:n0_tr)); te0 = idx0(r0(n0_tr+1:end));
tr1 = idx1(r1(1:n1_tr)); te1 = idx1(r1(n1_tr+1:end));

trainIdx_full = [tr0; tr1];
testIdx       = [te0; te1];

X_test = X_all(testIdx,:);
y_test = y_all(testIdx);

fprintf('[INFO] Train(full)=%d | Test=%d (Non-slip=%d, Slip=%d)\n', ...
    numel(trainIdx_full), numel(testIdx), sum(y_test==0), sum(y_test==1));

%% ========= 3) 验证集搜索 cap + C（可选阈值 thr） =========
bestValAcc = -inf;
best_cap = NaN; best_C = NaN; best_thr = 0; best_mode = 'ARGMAX';

logRows = {};  % 每行: cap, C, mode, thr, valAcc

for cap = trainCapCandidates
    [keepLocalIdx, ~] = localBalancedCapIndex(y_all(trainIdx_full), cap);
    trainIdx_cap = trainIdx_full(keepLocalIdx);
    y_cap = y_all(trainIdx_cap);

    fprintf('\n[INFO] === cap=%d -> nCap=%d ===\n', cap, numel(trainIdx_cap));

    [innerLocal, valLocal] = localStratifiedHoldout(y_cap, valRatio);
    idx_inner = trainIdx_cap(innerLocal);
    idx_val   = trainIdx_cap(valLocal);

    X_inner = X_all(idx_inner,:);  y_inner = y_all(idx_inner);
    X_val   = X_all(idx_val,:);    y_val   = y_all(idx_val);

    fprintf('[INFO] inner=%d, val=%d\n', numel(y_inner), numel(y_val));

    % 归一化：用 inner 拟合，再 apply 到 val
    [XtrN, ps] = mapminmax(double(X_inner)');   % K x Ninner
    XvaN = mapminmax('apply', double(X_val)', ps);

    traindatafile = [double(y_inner), XtrN'];
    valdatafile   = [double(y_val),   XvaN'];

    for C = C_candidates
        fprintf('    try C=%.4g ... ', C);

        try
            [~, ~, ~, TrainedMdl] = elm_kernel_train_withlabel(traindatafile, Elm_Type, C, Kernel_type, Kernel_para);
            [ValY, ~, ~] = elm_kernel_test_withlabel(TrainedMdl, valdatafile, testBlockSize);

            % ---- 强制 ValY = (nClass x Nval) ----
            nClass = size(TrainedMdl.OutputWeight,2);
            if size(ValY,1) ~= nClass && size(ValY,2) == nClass
                ValY = ValY';
            end

            % ---- ARGMAX ----
            label = TrainedMdl.Label(:)';   % [0 1]
            [~, predIdx] = max(ValY, [], 1);
            y_pred_argmax = label(predIdx).';
            acc_argmax = mean(double(y_pred_argmax(:)) == double(y_val(:)));

            acc_best = acc_argmax;
            thr_best = 0;
            mode_best = 'ARGMAX';

            % ---- THRESH（验证集上选阈值）----
            if enableThresholdTuning
                iNo = find(label==0,1); iSl = find(label==1,1);
                if ~isempty(iNo) && ~isempty(iSl)
                    scoreVal = (ValY(iSl,:) - ValY(iNo,:)).';
                    if numel(scoreVal) ~= numel(y_val)
                        error('scoreVal length (%d) != y_val length (%d). (ValY orientation issue)', ...
                              numel(scoreVal), numel(y_val));
                    end
                    [thrCand, accThr] = localTuneThreshold(scoreVal, y_val(:), nThrGrid);
                    if accThr > acc_best
                        acc_best = accThr;
                        thr_best = thrCand;
                        mode_best = 'THRESH';
                    end
                end
            end

            fprintf('valAcc=%.4f (%s)\n', acc_best, mode_best);
            logRows(end+1,:) = {cap, C, mode_best, thr_best, acc_best}; %#ok<AGROW>

            if acc_best > bestValAcc
                bestValAcc = acc_best;
                best_cap = cap; best_C = C; best_thr = thr_best; best_mode = mode_best;
            end

        catch ME
            % 这里不再只报一句“FAILED”，会把原因打印出来
            fprintf('FAILED (%s)\n', ME.message);
        end
    end
end

if ~isfinite(bestValAcc)
    error('All trials failed. Copy the FIRST FAILED message to me, I can pinpoint the exact line.');
end

fprintf('\n[BEST VAL] %.4f @ cap=%d, C=%.4g, mode=%s, thr=%.4g\n', bestValAcc, best_cap, best_C, best_mode, best_thr);

% 保存验证日志
Tlog = cell2table(logRows, 'VariableNames', {'cap','C','mode','thr','valAcc'});
writetable(Tlog, fullfile(directory_path, 'TuningLog_VAL_capC_thr.csv'));

%% ========= 4) 用最佳 cap/C 在 full capped-train 上重训，再 TEST =========
[keepLocalIdx2, ~] = localBalancedCapIndex(y_all(trainIdx_full), best_cap);
trainIdx_final = trainIdx_full(keepLocalIdx2);

X_train_final = X_all(trainIdx_final,:);
y_train_final = y_all(trainIdx_final);

[XtrN_final, ps_final] = mapminmax(double(X_train_final)');
XteN_final = mapminmax('apply', double(X_test)', ps_final);

traindatafile_final = [double(y_train_final), XtrN_final'];
testdatafile_final  = [double(y_test),        XteN_final'];

[~, TrainAcc, TrainTime, TrainedMdl_final] = ...
    elm_kernel_train_withlabel(traindatafile_final, Elm_Type, best_C, Kernel_type, Kernel_para);

[TestY, TestAcc_argmax, TestTime] = ...
    elm_kernel_test_withlabel(TrainedMdl_final, testdatafile_final, testBlockSize);

% ---- 强制 TestY = (nClass x Ntest) ----
nClass = size(TrainedMdl_final.OutputWeight,2);
if size(TestY,1) ~= nClass && size(TestY,2) == nClass
    TestY = TestY';
end

label = TrainedMdl_final.Label(:)'; % [0 1]
[~, predIdx] = max(TestY, [], 1);
y_pred_argmax = label(predIdx).';

y_pred_final = y_pred_argmax;
TestAcc_final = TestAcc_argmax;

if strcmp(best_mode,'THRESH')
    iNo = find(label==0,1); iSl = find(label==1,1);
    scoreTest = (TestY(iSl,:) - TestY(iNo,:)).';
    y_pred_thr = double(scoreTest > best_thr);
    y_pred_final = y_pred_thr;
    TestAcc_final = mean(double(y_pred_thr(:)) == double(y_test(:)));
end

fprintf('\n=== FINAL ===\n');
fprintf('TrainAcc=%.4f | TestAcc(ARGMAX)=%.4f | TestAcc(FINAL=%s)=%.4f\n', ...
    TrainAcc, TestAcc_argmax, best_mode, TestAcc_final);
fprintf('TrainTime=%.3fs | TestTime=%.3fs\n', TrainTime, TestTime);

%% ========= 5) 混淆矩阵 + 指标（窗口不关闭，方便截图） =========
classNames = {'Non-slip','Slip'};
order = [0 1];

cmTest = confusionmat(double(y_test), double(y_pred_final), 'Order', order);
metrics = localMetricsFromCM(cmTest);

disp('==== Test metrics (FINAL) ====');
disp(metrics.Table);
fprintf('Acc=%.4f | MacroF1=%.4f\n', metrics.Accuracy, metrics.MacroF1);

figPng = fullfile(directory_path, sprintf('ConfMat_TEST_SlipNoSlip_FINAL_%s.png', best_mode));
figFig = fullfile(directory_path, sprintf('ConfMat_TEST_SlipNoSlip_FINAL_%s.fig', best_mode));

fig = localPlotConfMatPercent(cmTest, classNames, ...
    sprintf('Test (cap=%d,K=%d,C=%.4g,mode=%s) Confusion Matrix (row-normalized %%)', best_cap, Num_feature, best_C, best_mode), ...
    figPng);
savefig(fig, figFig);

Tfinal = table(best_cap, Num_feature, best_C, string(best_mode), best_thr, TestAcc_final, ...
    metrics.Precision(2), metrics.Recall(2), metrics.F1(2), metrics.MacroF1, ...
    'VariableNames', {'cap','K','C','mode','thr','Accuracy','Precision_Slip','Recall_Slip','F1_Slip','MacroF1'});
writetable(Tfinal, fullfile(directory_path, 'SlipNoSlip_FINAL_TEST_metrics.csv'));

fprintf('[INFO] Saved PNG+FIG+CSV in %s\n', directory_path);
fprintf('[INFO] Figure stays OPEN for screenshot. Press any key in the figure to finish...\n');
waitforbuttonpress;

%% ======= local funcs =======
function [keepLocalIdx, yKeep] = localBalancedCapIndex(yTrain, totalCap)
    yTrain = double(yTrain(:));
    idx0 = find(yTrain==0); idx1 = find(yTrain==1);
    capEach = floor(totalCap/2);
    capEach = min([capEach, numel(idx0), numel(idx1)]);
    keep = [idx0(randperm(numel(idx0), capEach)); idx1(randperm(numel(idx1), capEach))];
    keep = keep(randperm(numel(keep)));
    keepLocalIdx = keep;
    yKeep = yTrain(keep);
end

function [idxTrain, idxVal] = localStratifiedHoldout(y, valRatio)
    y = double(y(:));
    idx0 = find(y==0); idx1 = find(y==1);
    n0v = max(1, round(numel(idx0)*valRatio));
    n1v = max(1, round(numel(idx1)*valRatio));
    p0 = randperm(numel(idx0)); p1 = randperm(numel(idx1));
    idxVal = [idx0(p0(1:n0v)); idx1(p1(1:n1v))];
    idxTrain = setdiff((1:numel(y))', idxVal, 'stable');
    idxVal = idxVal(randperm(numel(idxVal)));
    idxTrain = idxTrain(randperm(numel(idxTrain)));
end

function [thrBest, accBest] = localTuneThreshold(score, yTrue, nGrid)
    score = double(score(:));
    yTrue = double(yTrue(:));

    mn = min(score); mx = max(score);
    if mn == mx
        thrList = mn;
    else
        thrList = linspace(mn, mx, nGrid);
    end

    accBest = -inf; thrBest = 0;
    for t = thrList
        yPred = double(score > t);
        if numel(yPred) ~= numel(yTrue)
            error('Threshold tuning size mismatch: yPred=%d, yTrue=%d', numel(yPred), numel(yTrue));
        end
        acc = mean(yPred == yTrue);
        if acc > accBest
            accBest = acc; thrBest = t;
        end
    end
end

function M = localMetricsFromCM(cm)
    cm = double(cm);
    nClass = size(cm,1);
    total = sum(cm(:));
    acc = trace(cm) / max(total, 1);

    precision = zeros(nClass,1);
    recall    = zeros(nClass,1);
    f1        = zeros(nClass,1);

    for k = 1:nClass
        TP = cm(k,k);
        FP = sum(cm(:,k)) - TP;
        FN = sum(cm(k,:)) - TP;

        precision(k) = TP / max(TP + FP, 1e-12);
        recall(k)    = TP / max(TP + FN, 1e-12);

        if (precision(k) + recall(k)) > 0
            f1(k) = 2*precision(k)*recall(k)/(precision(k)+recall(k));
        else
            f1(k) = 0;
        end
    end

    M.Accuracy = acc;
    M.Precision = precision;
    M.Recall = recall;
    M.F1 = f1;
    M.MacroF1 = mean(f1);
    M.Table = table(precision, recall, f1, 'VariableNames', {'Precision','Recall','F1'});
end

function fig = localPlotConfMatPercent(cmCount, classNames, figTitle, savePngPath)
    cmCount = double(cmCount);
    rowSum = sum(cmCount,2); rowSum(rowSum==0) = 1;
    cmPct = 100 * (cmCount ./ rowSum);

    fig = figure('Color','w', 'Name', figTitle, 'NumberTitle','off');
    imagesc(cmPct); axis equal tight;

    colormap(localBlueMap(256)); caxis([0 100]);
    cb = colorbar; cb.Label.String = 'Row-normalized (%)';
    % 只保留 0/50/100 三个刻度
    if isprop(cb,'Ticks')   % 新版 MATLAB
        cb.Ticks = [0 50 100];
        cb.TickLabels = {'0','50','100'};  % 可选：强制显示成整数
    else                    % 兼容老版本
        set(cb,'YTick',[0 50 100],'YTickLabel',{'0','50','100'});
    end

    xticks(1:numel(classNames)); yticks(1:numel(classNames));
    xticklabels(classNames); yticklabels(classNames);
    xlabel('Predicted contact status'); ylabel('True contact status'); title(figTitle);
    set(gca,'FontName','Times New Roman','FontSize',16,'LineWidth',1.2);

    for i = 1:size(cmPct,1)
        for j = 1:size(cmPct,2)
            valPct = cmPct(i,j); valCnt = cmCount(i,j);
            txtColor = 'k'; if valPct >= 50, txtColor = 'w'; end
            text(j,i,sprintf('%.1f%%\n(%d)',valPct,valCnt), ...
                'HorizontalAlignment','center','VerticalAlignment','middle', ...
                'FontSize',15,'FontWeight','bold','Color',txtColor);
        end
    end

    try
        exportgraphics(fig, savePngPath, 'Resolution', 600);
    catch
        print(fig, savePngPath, '-dpng', '-r600');
    end
end

function cmap = localBlueMap(n)
    if nargin < 1, n = 256; end
    c1 = [0.92, 0.96, 0.99];
    c2 = [0.05, 0.25, 0.55];
    t = linspace(0,1,n).';
    cmap = (1-t).*c1 + t.*c2;
end
