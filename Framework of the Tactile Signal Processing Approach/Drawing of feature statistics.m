%% slipdetection_fs_v1_test6_pub_v8.m
% Publication-ready 2×2 summary figure (single-column) for Top-K features.
%
% Requested formatting (v6):
%   - Remove each subplot title and xlabel (keep tick labels)
%   - Use Times New Roman for ALL text
%   - Make tick labels as large as practical for single-column figures
%   - Reduce spacing between subplots (tiledlayout compact)
%   - Use different colors for each subplot; stacked subplots use unique colors
%   - Feature-type subplot x-tick labels rotated a bit more to save width

clear; clc; close all;

%% ---------------- User settings ----------------
TopK = 120;
modelFile = 'slip_classificationELMmodel.mat';   % change to full path if needed

% Figure settings for single-column papers (adjust if your template differs)
figWidthIn  = 3.45;   % ~8.8 cm (typical single-column)
figHeightIn = 4.55;   % a bit taller to keep large tick labels inside the exported frame   % slightly taller to accommodate large tick labels
fontName    = 'Times New Roman';

tickFont    = 18;     % tick label font size (very large)
yLabelFont  = 20;     % y-axis label font size
legendFont  = 16;     % legend font size
barTargetPx = 20;     % target bar thickness in pixels (equalize across subplots)
exportPad  = 10;     % extra padding (points) to prevent tick labels being clipped on export

% Colors (ALL UNIQUE across the whole figure)
% Feature-type (single color)
cFeature = [0.0000 0.4470 0.7410];   % blue
% Band-type (single color)
cBand    = [0.4940 0.1840 0.5560];   % purple
% SE-type stacked colors (Domain: T, F) -- unique colors
cDomT    = [0.4660 0.6740 0.1880];   % green
cDomF    = [0.9290 0.6940 0.1250];   % yellow
% Finger-type stacked colors (PVDF, SG) -- unique colors
cPVDF    = [0.3010 0.7450 0.9330];   % cyan
cSG      = [0.8500 0.3250 0.0980];   % orange

% --- Finger subplot: use your provided distribution (RECOMMENDED) ---
useManualFingerCounts = true;
manualFingerLabels = ["Little finger","Ring finger","Middle finger","Index finger","Thumb"];
manualFinger_SG    = [0, 2, 4, 0, 0];
manualFinger_PVDF  = [15,21,17,48,13];
 
% --- Feature-type (subplot d) x-tick label customization ---
% You can rename the Feature-type tick labels without changing the underlying category names.
% Option A (recommended): provide a 2-column string array mapping:
%   [ "OriginalName1"  "DisplayLabel1"
%     "OriginalName2"  "DisplayLabel2" ]
% Any feature type not listed keeps its original name.
%
% Example:
% featureTypeLabelMap = [
%     "mean"   "Mean"
%     "std"    "Std"
%     "skew"   "Skewness"
% ];
featureTypeLabelMap = strings(0,2);   % <-- EDIT HERE if you want renaming

% Option B: directly specify ALL labels in order (must match all feature types shown, e.g., 17).
% If you enable this, it overrides Option A.
useFeatureTypeLabelsByOrder = true;  % set true to use the array below
featureTypeLabelsByOrder = [
    "Peak factor"
    "Kurtosis"
    "Skewness"
    "Spectral centroid"
    "Band energy ratio"
    "Coefficient of variation"
    "Standard Deviation"
    "Range"
    "Entropy"
    "Mean deviation"
    "Interquartile range"
    "Median"
    "Mean"
    "Spectral spread"
    "Correlation coefficient"
    "Energy"
    "Mode"

];  % e.g., ["L1";"L2";...;"L17"]  (column vector)
printFeatureTypeLabelList = false;  % set true to print original->display label list in Command Window

assert(exist(modelFile,'file')==2, "Cannot find %s", modelFile);

%% ---------------- Set global fonts (restore on exit) ----------------
root = groot;
oldDefaults = struct();
propList = { ...
    'defaultAxesFontName', 'defaultTextFontName', 'defaultLegendFontName', ...
    'defaultAxesFontSize', 'defaultTextFontSize' ...
    };
for i = 1:numel(propList)
    p = propList{i};
    try
        oldDefaults.(p) = get(root, p);
    catch
        % Some properties may not exist in older releases; ignore.
    end
end

cleanupObj = onCleanup(@() restoreRootDefaults(root, oldDefaults));

try
    set(root, 'defaultAxesFontName',   fontName);
    set(root, 'defaultTextFontName',   fontName);
    set(root, 'defaultLegendFontName', fontName);
catch
end

%% ---------------- Load ----------------
S = load(modelFile);
assert(isfield(S,'Featureinforsorted'), "model file missing Featureinforsorted");

FIS = S.Featureinforsorted;
TopK = min(TopK, numel(FIS.name));

FeatureName = cleanStr(string(FIS.name(1:TopK)));
Stat        = cleanStr(string(FIS.fname(1:TopK)));
Band        = cleanStr(string(FIS.band(1:TopK)));
Domain      = cleanStr(string(FIS.domain(1:TopK)));    % e.g., 'T' or 'F'
SensorSE    = cleanStr(string(FIS.SE(1:TopK)));        % e.g., 'SG' or 'PVDF'

%% ---------------- (a) Sensor type × Domain ----------------
desiredSensors = ["PVDF","SG"];
desiredDomains = ["Time","Frequency"];   % show both even if some are missing

sensorCats = localOrderedCats(SensorSE, desiredSensors);
domainCats = localOrderedCats(Domain,   desiredDomains);

SD = zeros(numel(sensorCats), numel(domainCats));
for i = 1:numel(sensorCats)
    for j = 1:numel(domainCats)
        SD(i,j) = sum(SensorSE == sensorCats(i) & Domain == domainCats(j));
    end
end

%% ---------------- (b) Feature type (Stat) ----------------
% Show ALL feature types (e.g., 17 types) even if some types have 0 count in TopK,
% to keep the figure rigorous.
%
% We build a "master" feature-type list from *all* features in Featureinforsorted,
% then count TopK using that list so missing types appear with 0 bars.

AllStat = cleanStr(string(FIS.fname(:)));              % all feature types in the model
allStatTypes = unique(AllStat, 'stable');             % keep original order ('stable')

% Drop placeholder "Unknown" if it only comes from empty strings and is not in TopK.
if any(allStatTypes=="Unknown") && ~any(Stat=="Unknown")
    allStatTypes(allStatTypes=="Unknown") = [];
end

% Order: types present in TopK (sorted by count) first, then the remaining types (0-count).
tmpTop    = localCountBy(Stat, []);                   % sorted desc by Count (TopK only)
topOrder  = tmpTop.Category;
statOrder = [topOrder; setdiff(allStatTypes, topOrder, 'stable')];

StatCount = localCountBy(Stat, statOrder);

% Build displayed tick labels for the Feature-type subplot (can be customized above).
StatTickLabels = string(StatCount.Category(:));   % default: original names

% Option A: mapping original -> display
if exist('featureTypeLabelMap','var') && ~isempty(featureTypeLabelMap)
    assert(size(featureTypeLabelMap,2)==2, ...
        'featureTypeLabelMap must be an N×2 string array: ["orig","display"; ...].');
    for rr = 1:size(featureTypeLabelMap,1)
        k = string(featureTypeLabelMap(rr,1));
        v = string(featureTypeLabelMap(rr,2));
        StatTickLabels(StatTickLabels == k) = v;
    end
end

% Option B: explicit labels by order (overrides mapping)
if exist('useFeatureTypeLabelsByOrder','var') && useFeatureTypeLabelsByOrder
    lbl = string(featureTypeLabelsByOrder(:));
    assert(numel(lbl) == numel(StatTickLabels), ...
        'featureTypeLabelsByOrder must have %d labels (one for each feature type shown).', numel(StatTickLabels));
    StatTickLabels = lbl;
end

% Optional: print label list for easy editing
if exist('printFeatureTypeLabelList','var') && printFeatureTypeLabelList
    disp('Feature type tick labels (Original -> Displayed):');
    disp(table(string(StatCount.Category(:)), string(StatTickLabels(:)), ...
        'VariableNames', {'Original','Displayed'}));
end

%% ---------------- (c) Band ----------------
% Always show D1 even if 0.
preferredBand = ["raw","A1","A2","A3","A4","D1","D2","D3","D4"];
BandCount = localCountBy(Band, preferredBand);

%% ---------------- (d) Finger ----------------
if useManualFingerCounts
    fingerCats = string(manualFingerLabels(:));
    fingerCats = fingerCats(:);
    fingerY = [manualFinger_PVDF(:), manualFinger_SG(:)];   % N×2
    FingerCount = table(fingerCats, sum(fingerY,2), 'VariableNames', {'Category','Count'});
    assert(sum(FingerCount.Count) == TopK, ...
        'Manual finger counts sum to %d, but TopK=%d. Please check manualFinger_* arrays.', ...
        sum(FingerCount.Count), TopK);
else
    sensorId = extractSensorId(FeatureName);
    Finger   = mapSensorToFinger(sensorId);
    fingerOrder = ["Little finger","Ring finger","Middle finger","Index finger","Thumb","Unknown"];
    FingerCount = localCountBy(Finger, fingerOrder);
    if any(FingerCount.Category == "Unknown") && all(FingerCount.Count(FingerCount.Category=="Unknown") == 0)
        FingerCount(FingerCount.Category=="Unknown",:) = [];
    end
    fingerCats = FingerCount.Category;
    fingerY = [FingerCount.Count, zeros(size(FingerCount.Count))];
end

%% ---------------- Build 2×2 figure ----------------
f = figure('Color','w','Units','inches','Position',[1 1 figWidthIn figHeightIn]);

useTiled = (exist('tiledlayout','file') == 2) && (exist('nexttile','file') == 2);
if useTiled
    t = tiledlayout(f, 2, 2, 'TileSpacing','tight', 'Padding','tight');
    % Leave a bit of bottom margin so large/rotated tick labels (subplot d) stay inside the frame
    try
        t.Units = 'normalized';
        t.OuterPosition = [0.02 0.10 0.96 0.88];
    catch
    end
else
    t = []; %#ok<NASGU>
end

% Helper for axes creation
ax1 = nextTileCompat(1, useTiled);
ax2 = nextTileCompat(2, useTiled);
ax3 = nextTileCompat(3, useTiled);
ax4 = nextTileCompat(4, useTiled);

% Swap tiles 2 and 4 (right column) so Band is top-right and Feature is bottom-right
% Tile indices follow nexttile indexing for a 2x2 layout. See MathWorks doc: nexttile.
% https://www.mathworks.com/help/matlab/ref/nexttile.html
tmp = ax2; ax2 = ax4; ax4 = tmp;
clear tmp;


% ---- (d) Finger (stacked: PVDF vs SG)  [top-left]
axes(ax1); %#ok<LAXES>
X4 = categorical(fingerCats, fingerCats, fingerCats);
b4 = bar(ax1, X4, fingerY, 'stacked');
styleAxes(ax1, tickFont, fontName);
ylabel(ax1, 'Count', 'FontSize', yLabelFont, 'FontName', fontName);
grid(ax1, 'on'); box(ax1, 'on');
xtickangle(ax1, 45);
% Colors (unique)
set(b4(1), 'FaceColor', cPVDF, 'EdgeColor', 'k', 'LineWidth', 0.6);
set(b4(2), 'FaceColor', cSG,   'EdgeColor', 'k', 'LineWidth', 0.6);
lg4 = legend(ax1, {'PVDF','SG'}, 'Location','best');
set(lg4, 'FontSize', legendFont, 'FontName', fontName);

% ---- (b) Feature type  [bottom-right]
X2 = categorical(StatCount.Category, StatCount.Category, StatCount.Category);
b2 = bar(ax2, X2, StatCount.Count);
styleAxes(ax2, tickFont, fontName);
ylabel(ax2, 'Count', 'FontSize', yLabelFont, 'FontName', fontName);
grid(ax2, 'on'); box(ax2, 'on');
xtickangle(ax2, 65);  % larger angle to save horizontal space
set(b2, 'FaceColor', cFeature, 'EdgeColor', 'k', 'LineWidth', 0.6);
% Apply custom x-tick labels (display only; data categories remain unchanged)
xticklabels(ax2, cellstr(StatTickLabels));

% ---- (a) SE type × Domain (stacked)  [bottom-left]
X1 = categorical(sensorCats, sensorCats, sensorCats);
b1 = bar(ax3, X1, SD, 'stacked');
styleAxes(ax3, tickFont, fontName);
ylabel(ax3, 'Count', 'FontSize', yLabelFont, 'FontName', fontName);
grid(ax3, 'on'); box(ax3, 'on');
% Domain colors (unique)
% b1(i) corresponds to column i in SD, i.e., domainCats(i)
for ii = 1:numel(b1)
    try
        if domainCats(ii) == "Time"
            set(b1(ii), 'FaceColor', cDomT, 'EdgeColor', 'k', 'LineWidth', 0.6);
        elseif domainCats(ii) == "Frequency"
            set(b1(ii), 'FaceColor', cDomF, 'EdgeColor', 'k', 'LineWidth', 0.6);
        else
            % Fallback (should not happen)
            set(b1(ii), 'EdgeColor', 'k', 'LineWidth', 0.6);
        end
    catch
    end
end
lg1 = legend(ax3, cellstr(domainCats), 'Location','best');
set(lg1, 'FontSize', legendFont, 'FontName', fontName);

% ---- (c) Band type  [top-right]
X3 = categorical(BandCount.Category, BandCount.Category, BandCount.Category);
b3 = bar(ax4, X3, BandCount.Count);
styleAxes(ax4, tickFont, fontName);
ylabel(ax4, 'Count', 'FontSize', yLabelFont, 'FontName', fontName);
grid(ax4, 'on'); box(ax4, 'on');
set(b3, 'FaceColor', cBand, 'EdgeColor', 'k', 'LineWidth', 0.6);

% ---- Remove subplot titles & xlabels (keep ticks) ----
removeTitleXLabel(ax1);
removeTitleXLabel(ax2);
removeTitleXLabel(ax3);
removeTitleXLabel(ax4);

% Prevent underscores from becoming subscripts in tick labels
set([ax1,ax2,ax3,ax4], 'TickLabelInterpreter','none');

% Compute layout and axes pixel sizes
drawnow;

% Reduce extra margins around axes (helps tighten layout)
tightenAxesMargins(ax1);
tightenAxesMargins(ax2);
tightenAxesMargins(ax3);
tightenAxesMargins(ax4);

% Equalize bar thickness across subplots using a pixel target
applyPixelBarWidth(ax1, b4, numel(fingerCats), barTargetPx);
applyPixelBarWidth(ax2, b2, height(StatCount), barTargetPx);
applyPixelBarWidth(ax3, b1, numel(sensorCats), barTargetPx);
applyPixelBarWidth(ax4, b3, height(BandCount), barTargetPx);

%% ---------------- Sanity checks ----------------
fprintf('\n===== Sanity checks =====\n');
fprintf('TopK = %d\n', TopK);
fprintf('Sum(Feature-type counts) = %d\n', sum(StatCount.Count));
fprintf('Sum(Band counts)         = %d\n', sum(BandCount.Count));
fprintf('Sum(Finger counts)       = %d\n', sum(FingerCount.Count));
fprintf('Sum(Sensor×Domain counts)= %d\n', sum(SD,'all'));

%% ---------------- Export ----------------
outBase = sprintf('Top%d_4plots_2x2_pub', TopK);

if exist('exportgraphics','file') == 2
    if useTiled
        exportgraphics(t, outBase + ".png", 'Resolution', 600, 'Padding', exportPad);
        exportgraphics(t, outBase + ".pdf", 'ContentType', 'vector', 'Padding', exportPad);
    else
        exportgraphics(f, outBase + ".png", 'Resolution', 600, 'Padding', exportPad);
        exportgraphics(f, outBase + ".pdf", 'ContentType', 'vector', 'Padding', exportPad);
    end
else
    try, set(f,'PaperPositionMode','auto'); catch, end
    print(f, outBase + ".png", '-dpng', '-r600');
    print(f, outBase + ".pdf", '-dpdf');
end

disp('Done.');

%% ===================== Local functions =====================

function restoreRootDefaults(root, oldDefaults)
    fns = fieldnames(oldDefaults);
    for i = 1:numel(fns)
        try
            set(root, fns{i}, oldDefaults.(fns{i}));
        catch
        end
    end
end

function removeTitleXLabel(ax)
% Remove title and xlabel but keep tick labels.
    try
        ax.Title.String = '';
        ax.Title.Visible = 'off';
    catch
    end
    try
        ax.XLabel.String = '';
        ax.XLabel.Visible = 'off';
    catch
    end
end

function tightenAxesMargins(ax)
% Reduce extra whitespace around the axes.
    try
        ax.LooseInset = ax.TightInset;
    catch
    end
end

function s = cleanStr(s)
% Trim and replace missing/empty strings with "Unknown".
    s = strip(string(s(:)));
    s(ismissing(s) | s=="") = "Unknown";
end

function cats = localOrderedCats(x, desiredFirst)
% Return category list where desiredFirst (if present) goes first.
    x = cleanStr(x);
    desiredFirst = string(desiredFirst(:));
    found = unique(x, 'stable');
    found = found(:); % ensure column for safe vertical concatenation
    firstPart  = desiredFirst(ismember(desiredFirst, found));
    firstPart  = firstPart(:);
    secondPart = setdiff(found, desiredFirst, 'stable');
    secondPart = secondPart(:);
    cats = [firstPart; secondPart];
end

function T = localCountBy(x, order)
% Count occurrences of string array x.
% If order is provided, keep that order and include zero-count categories
% for items in order even if they don't appear in x.
    x = cleanStr(x);
    if nargin < 2, order = []; end

    if isempty(order)
        cats = unique(x, 'stable');
        cats = cats(:);
        C = categorical(x, cats, cats);
        cnt = countcats(C);
        total = sum(cnt);
        if total == 0
            pct = zeros(size(cnt));
        else
            pct = 100 * cnt / total;
        end
        T = table(string(cats(:)), cnt(:), pct(:), 'VariableNames', {'Category','Count','Percent'});
        T = sortrows(T, 'Count', 'descend');
    else
        order = string(order(:));
        found = unique(x, 'stable');
        found = found(:);
        extra = setdiff(found, order, 'stable');
        extra = extra(:);
        cats = [order(:); extra];
        C = categorical(x, cats, cats);
        cnt = countcats(C);
        total = sum(cnt);
        if total == 0
            pct = zeros(size(cnt));
        else
            pct = 100 * cnt / total;
        end
        T = table(string(cats(:)), cnt(:), pct(:), 'VariableNames', {'Category','Count','Percent'});
    end
end

function ax = nextTileCompat(k, useTiled)
% Create axes in a tiledlayout if available; otherwise use subplot.
    if useTiled
        ax = nexttile(k);
    else
        ax = subplot(2,2,k);
    end
end

function styleAxes(ax, tickFont, fontName)
    set(ax, 'FontSize', tickFont, 'FontName', fontName, ...
        'LineWidth', 0.9, 'TickDir','out');
end

function applyPixelBarWidth(ax, b, nCats, targetPx)
% Set BarWidth based on axes pixel width so bars look equally thick across subplots.
% BarWidth is a fraction of the category spacing.
    if nCats <= 0
        return;
    end
    pos = getpixelposition(ax, true);
    axWidthPx = pos(3);
    catPx = axWidthPx / nCats;
    bw = targetPx / catPx;
    bw = max(0.05, min(0.9, bw));
    for i = 1:numel(b)
        try
            b(i).BarWidth = bw;
        catch
        end
    end
end

function sid = extractSensorId(featureNames)
% Extract sensor id like "SG7" or "PVDF10" from feature name strings.
    featureNames = string(featureNames(:));
    sid = strings(numel(featureNames),1);
    for i = 1:numel(featureNames)
        m = regexp(featureNames(i), '(SG\d+|PVDF\d+)', 'match', 'once');
        if isempty(m)
            sid(i) = "Unknown";
        else
            sid(i) = string(m);
        end
    end
end

function finger = mapSensorToFinger(sensorId)
% Map each sensor ID to a finger group.
    sensorId = string(sensorId(:));
    finger = repmat("Unknown", numel(sensorId), 1);

    little = ["SG1","SG2","SG3","PVDF1","PVDF2"];
    ring   = ["SG4","SG5","SG6","PVDF3","PVDF4"];
    middle = ["SG7","SG8","SG9","PVDF5","PVDF6"];
    index  = ["SG10","SG11","SG12","PVDF7","PVDF8"];
    thumb  = ["SG13","SG14","PVDF9","PVDF10"];

    finger(ismember(sensorId, little)) = "Little finger";
    finger(ismember(sensorId, ring))   = "Ring finger";
    finger(ismember(sensorId, middle)) = "Middle finger";
    finger(ismember(sensorId, index))  = "Index finger";
    finger(ismember(sensorId, thumb))  = "Thumb";
end