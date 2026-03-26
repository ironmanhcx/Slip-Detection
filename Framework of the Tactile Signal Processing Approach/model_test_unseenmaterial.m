% model_test_unseenmaterial_test3_fix_nomem.m
% -------------------------------------------------------------------------
% 解决你遇到的报错：
%   Out of memory @ elm_kernel_test>kernel_matrix:
%     omega = (Xtrain*Xt'+c).^d
% 原因：
%   elm_kernel_test 会一次性构造 Omega_test (Ntrain x Ntest) 的核矩阵。
%   对你的数据规模 (Ntrain 很大，Ntest 也很大) 会直接爆内存。
%
% 本文件做的改进（保持你原来的流程、结果一致，但不爆内存）：
%   1) 不再直接调用 elm_kernel_test（它会一次性生成超大 Omega_test）
%   2) 改为“分块(block)计算核矩阵 + 分块得到输出”，显著降低峰值内存
%   3) 继续输出论文用：Slip/No-slip 混淆矩阵（白->深蓝 + colorbar）
%      并计算 Accuracy/Precision/Recall/F1（Slip=1 为正类）
%
% MATLAB 官方文档（老师要求：不确定就查）：
%   - 解决 Out of Memory: https://ww2.mathworks.cn/help/matlab/matlab_prog/resolving-out-of-memory-errors.html
%   - memory (Windows):    https://www.mathworks.com/help/matlab/ref/memory.html
%   - matfile:             https://www.mathworks.com/help/matlab/ref/matlab.io.matfile.html
%   - confusionmat:        https://www.mathworks.com/help/stats/confusionmat.html
%   - exportgraphics:      https://www.mathworks.com/help/matlab/ref/exportgraphics.html
% -------------------------------------------------------------------------

clc;
clear;
close all;

%% ====================== 1) 路径与文件选择 ======================
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);

directory_path = thisDir;   % <-- 如数据在别处，请改成你的路径

slipFeatFile = fullfile(directory_path, 'SlipBinnedFeature.mat');
modelFile    = fullfile(directory_path, 'slip_classificationELMmodel.mat');

if ~isfile(slipFeatFile)
    [f,p] = uigetfile('*.mat','请选择 SlipBinnedFeature.mat');
    if isequal(f,0); error('未选择 SlipBinnedFeature.mat，无法继续。'); end
    slipFeatFile = fullfile(p,f);
    directory_path = p;
end

if ~isfile(modelFile)
    [f,p] = uigetfile('*.mat','请选择 slip_classificationELMmodel.mat');
    if isequal(f,0); error('未选择 slip_classificationELMmodel.mat，无法继续。'); end
    modelFile = fullfile(p,f);
end

%% ====================== 2) 加载模型（训练好的 6M 模型） ======================
Smdl = load(modelFile, 'TrainedMdl', 'inputps', 'Num_feature', 'idxfeature', ...
                     'TrainingAccuracy', 'TestingAccuracy');
TrainedMdl  = Smdl.TrainedMdl;
inputps     = Smdl.inputps;
Num_feature = Smdl.Num_feature;
idxfeature  = Smdl.idxfeature(:);

TrainingAccuracy_6M = NaN;
TestingAccuracy_6M  = NaN;
if isfield(Smdl,'TrainingAccuracy'); TrainingAccuracy_6M = Smdl.TrainingAccuracy; end
if isfield(Smdl,'TestingAccuracy');  TestingAccuracy_6M  = Smdl.TestingAccuracy;  end

%% ====================== 3) 加载 R4M 测试集（尽量只读 Top-K 列） ======================
% 为了进一步省内存：优先用 matfile 只读取 featurepoolR4M 的 Top-K 列
idxTop = idxfeature(1:Num_feature);

try
    m = matfile(slipFeatFile);  % 要求 SlipBinnedFeature.mat 为 -v7.3 才能按列索引不全读
    yTest = double(m.output_srclassR4M(:));
    Xtest_raw = m.featurepoolR4M(:, idxTop);   % 只读 Top-K 特征
catch
    % 如果不是 v7.3 或无法按列索引，则退化为普通 load（可能更吃内存）
    Sfeat = load(slipFeatFile, 'featurepoolR4M', 'output_srclassR4M');
    yTest = double(Sfeat.output_srclassR4M(:));
    Xtest_raw = Sfeat.featurepoolR4M(:, idxTop);
end

testnum = size(Xtest_raw, 1);

%% ====================== 4) 归一化（用训练时 inputps）并构造 testdatafile ======================
Xtest_T = Xtest_raw'; % d x N
Xtest_norm_T = mapminmax('apply', Xtest_T, inputps);
Xtest_norm = Xtest_norm_T'; % N x d

testdatafile = [yTest, Xtest_norm]; % elm 输入格式：第一列标签

%% ====================== 5) 分块 Kernel-ELM 测试（避免一次性大核矩阵） ======================
% 关键参数：blockSize 越小越省内存，但越慢。建议 200~800 之间试。
blockSize = 400;

[TestY, yPred, TestingAccuracy_unseen, TestingTime] = local_elm_kernel_test_blocked(TrainedMdl, testdatafile, blockSize); %#ok<ASGLU>

%% ====================== 6) 混淆矩阵 + 指标（Slip=1 为正类） ======================
yTrue = yTest(:);

order = [0 1]; % No-slip(0), Slip(1)
C = confusionmat(yTrue, yPred, 'Order', order);

TN = C(1,1); FP = C(1,2);
FN = C(2,1); TP = C(2,2);

Accuracy  = (TP + TN) / max(1, sum(C,'all'));
Precision = TP / max(1, (TP + FP));
Recall    = TP / max(1, (TP + FN));
F1        = 2*Precision*Recall / max(1e-12, (Precision + Recall));

%% ====================== 7) 画“准确率/召回率风格”的混淆矩阵（行归一化） ======================
rowSum = sum(C,2);
Cn = zeros(size(C));
for r = 1:size(C,1)
    if rowSum(r) > 0
        Cn(r,:) = C(r,:) / rowSum(r);
    end
end

fig = figure('Color','w','Units','pixels','Position',[120 120 900 650]);
Cn_plot = 100 * Cn;  % 用百分比(0~100)显示
imagesc(Cn_plot);
axis equal tight;
set(gca,'XTick',1:2,'YTick',1:2);
set(gca,'XTickLabel',{'Non-slip','Slip'}, 'YTickLabel',{'Non-slip','Slip'});
xlabel('Predicted contact status');
ylabel('True contact status');
title(sprintf('', ...
    100*Accuracy, Precision, Recall, F1), 'FontWeight','bold');

set(gca,'FontName','Times New Roman','FontSize',22,'LineWidth',1.2);

colormap(localWhiteToDeepBlue(256));
caxis([0 100]);
cb = colorbar;
cb.Ticks = [0 50 100];
cb.Label.String = 'Row-normalized (%)';

for r = 1:2
    for c = 1:2
        pct = Cn_plot(r,c);
        cnt = C(r,c);
        txt = sprintf('%.1f%%\n(%d)', pct, cnt);
        if Cn_plot(r,c) > 50
            tcolor = 'w';
        else
            tcolor = 'k';
        end
        text(c, r, txt, 'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
            'FontName','Times New Roman', 'FontSize',22, 'FontWeight','bold', 'Color',tcolor);
    end
end
box on;

%% ====================== 8) 导出图片 + 保存结果 ======================
outPrefix = fullfile(directory_path, 'ConfMat_UnseenR4M_SlipNoSlip');

try
    exportgraphics(fig, [outPrefix '.png'], 'Resolution', 600);
    exportgraphics(fig, [outPrefix '.pdf'], 'ContentType', 'vector');
catch
    print(fig, [outPrefix '.png'], '-dpng', '-r600');
    print(fig, [outPrefix '.pdf'], '-dpdf');
end

fprintf('\n==================== Accuracy Summary ====================\n');
if ~isnan(TrainingAccuracy_6M)
    fprintf('TrainingAccuracy (6M model file) = %.4f\n', TrainingAccuracy_6M);
end
if ~isnan(TestingAccuracy_6M)
    fprintf('TestingAccuracy  (6M model file) = %.4f\n', TestingAccuracy_6M);
end
fprintf('TestingAccuracy  (Unseen R4M)     = %.4f\n', TestingAccuracy_unseen);
fprintf('TestingTime (blocked)            = %.3fs (blockSize=%d)\n', TestingTime, blockSize);

fprintf('\n==================== Metrics (Slip=Positive) ==============\n');
fprintf('Accuracy  = %.4f\n', Accuracy);
fprintf('Precision = %.4f\n', Precision);
fprintf('Recall    = %.4f\n', Recall);
fprintf('F1        = %.4f\n', F1);
fprintf('TP=%d, FP=%d, FN=%d, TN=%d\n', TP, FP, FN, TN);

output = struct();
output.expected = yTrue;
output.produced = yPred;

metricsTbl = table(Accuracy, Precision, Recall, F1, TP, FP, FN, TN);

save(fullfile(directory_path,'slip_classificationELMunseen_withCM.mat'), ...
    'TestingAccuracy_unseen', 'TrainingAccuracy_6M', 'TestingAccuracy_6M', ...
    'output', 'C', 'Cn', 'metricsTbl', 'blockSize');

writetable(metricsTbl, fullfile(directory_path,'SlipNoSlip_Metrics_UnseenR4M.csv'));

fprintf('\n[Saved] Confusion matrix figure: %s(.png/.pdf)\n', outPrefix);
fprintf('[Saved] MAT results: %s\n', fullfile(directory_path,'slip_classificationELMunseen_withCM.mat'));
fprintf('[Saved] CSV metrics: %s\n', fullfile(directory_path,'SlipNoSlip_Metrics_UnseenR4M.csv'));

%% ====================== Local Functions ======================

function [TestY, yPred, TestingAccuracy, TestingTime] = local_elm_kernel_test_blocked(TrainedMdl, TestingData, blockSize)
% 功能：等价于 elm_kernel_test，但用 block 方式计算 Omega_test，避免内存爆炸

REGRESSION = 0;
CLASSIFIER = 1;

P = TrainedMdl.P;  % d x Ntrain
Kernel_type = TrainedMdl.Kernel_type;
Kernel_para = TrainedMdl.Kernel_para;
OutputWeight = TrainedMdl.OutputWeight; % Ntrain x nOut
Elm_Type = TrainedMdl.Elm_Type;

test_data = TestingData;
TV_T = test_data(:,1);           % Ntest x 1
TV_P = test_data(:,2:end)';      % d x Ntest
clear test_data;

Ntest = size(TV_P,2);
nOut  = size(OutputWeight,2);

% 预分配输出（2 x Ntest 很小，不会爆）
TestY = zeros(nOut, Ntest);

tic;

% 自动兜底：如果 blockSize 还是大导致 nomem，就自动减半重试
maxRetry = 6;
retry = 0;

while true
    try
        % 主循环：按 block 计算核矩阵和输出
        for s = 1:blockSize:Ntest
            e = min(Ntest, s + blockSize - 1);
            Xt_block = TV_P(:,s:e)'; % (nBlock x d)

            Omega_block = local_kernel_matrix(P', Kernel_type, Kernel_para, Xt_block); % (Ntrain x nBlock)
            TY_block = (Omega_block' * OutputWeight)'; % (nOut x nBlock)

            TestY(:,s:e) = TY_block;
        end
        break; % 成功退出
    catch ME
        if contains(lower(ME.message), 'out of memory') || strcmp(ME.identifier,'MATLAB:nomem')
            retry = retry + 1;
            if retry > maxRetry || blockSize <= 50
                rethrow(ME);
            end
            blockSize = max(50, floor(blockSize/2));
            % 清空已经算的部分，重新来（保证逻辑一致）
            TestY(:) = 0;
            continue;
        else
            rethrow(ME);
        end
    end
end

TestingTime = toc;

% 预测标签：取 argmax
[~, predIdx] = max(TestY, [], 1);  % 1..nOut
predIdx = predIdx(:);

% 将类别索引映射回原始标签值（这里默认训练标签是 [0 1]）
% 你的 slip/no-slip 是 0/1，所以可直接 predIdx-1
yPred = predIdx - 1;

% TestingAccuracy（自己算，避免依赖 elm_kernel_test 的 one-hot 逻辑）
if Elm_Type == REGRESSION
    TestingAccuracy = sqrt(mse(TV_T' - TestY));
else
    yTrue = TV_T(:);
    TestingAccuracy = 1 - mean(yPred ~= yTrue);
end
end

function omega = local_kernel_matrix(Xtrain, kernel_type, kernel_pars, Xt)
% Xtrain: Ntrain x d
% Xt:     NtestBlock x d
% omega:  Ntrain x NtestBlock

nb_data = size(Xtrain,1);

if strcmp(kernel_type,'RBF_kernel')
    XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
    XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
    omega = XXh1+XXh2' - 2*(Xtrain*Xt');
    omega = exp(-omega./kernel_pars(1));

elseif strcmp(kernel_type,'lin_kernel')
    omega = Xtrain*Xt';

elseif strcmp(kernel_type,'poly_kernel')
    omega = (Xtrain*Xt'+kernel_pars(1)).^kernel_pars(2);

elseif strcmp(kernel_type,'wav_kernel')
    XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
    XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
    omega = XXh1+XXh2' - 2*(Xtrain*Xt');

    XXh11 = sum(Xtrain,2)*ones(1,size(Xt,1));
    XXh22 = sum(Xt,2)*ones(1,nb_data);
    omega1 = XXh11-XXh22';

    omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
else
    error('Unknown kernel type: %s', kernel_type);
end
end

function cmap = localWhiteToDeepBlue(n)
% 生成白色 -> 深蓝色渐变色（n x 3）
deepBlue = [0.0, 0.2, 0.8];
t = linspace(0,1,n)';
cmap = (1-t).*ones(1,3) + t.*deepBlue;
end
