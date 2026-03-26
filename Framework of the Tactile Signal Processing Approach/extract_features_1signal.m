function [featRow, info] = extract_features_1signal(x, sensorType, fs, wavLevels, domainFlag, wavName)
%EXTRACT_FEATURES_1SIGNAL
% 通用单通道特征提取：
%   输入：  x(1xN) 或 (Nx1) 信号片段
%          sensorType: 'pvdf'/'sg'（也允许 'pvdf1','pvdf2','sg1','sg2'）
%          fs: 采样率
%          wavLevels: 小波分解层数（0 表示不做小波）
%          domainFlag: 'T' / 'F' / 'TF'
%          wavName: 小波名称（可省略，默认 'rbio2.2'）
%
%   输出：  featRow: 1xD 特征行向量
%          info: 结构体，包含 featNames/band/domain/statName 以及小波/FFT信息
%


    if nargin < 6 || isempty(wavName)
        wavName = 'rbio2.2';
    end
    if nargin < 5
        error('extract_features_1signal:MissingInputs', '需要至少 5 个输入参数。');
    end

    % --------- 规范化输入 ---------
    x = double(x(:))';               % 强制为 1xN
    N = numel(x);
    if N < 2
        error('extract_features_1signal:SignalTooShort', '信号长度太短（N<2）。');
    end

    sensorTypeStr = char(sensorType);
    sensorTypeLow = lower(sensorTypeStr);

    % 将 'pvdf1'/'pvdf2' 也视作 pvdf；'sg1'/'sg2' 也视作 sg
    if startsWith(sensorTypeLow, 'pvdf')
        family = 'pvdf';
    elseif startsWith(sensorTypeLow, 'sg')
        family = 'sg';
    else
        error('extract_features_1signal:UnknownSensorType', ...
            'sensorType 必须以 pvdf* 或 sg* 开头（例如 pvdf, pvdf1, sg, sg2）。');
    end

    dom = upper(string(domainFlag));
    if ~(dom=="T" || dom=="F" || dom=="TF")
        error('extract_features_1signal:BadDomainFlag', "domainFlag 必须是 'T'/'F'/'TF'。");
    end
    useT = (dom=="T" || dom=="TF");
    useF = (dom=="F" || dom=="TF");

    % --------- 输出容器（动态拼接） ---------
    feat = [];
    featNames = {};
    bandList  = {};
    domList   = {};
    statList  = {};

    % --------- info 基本信息 ---------
    info = struct();
    info.sensorType  = sensorTypeStr;   
    info.family      = family;          % 'pvdf' or 'sg'
    info.fs          = fs;
    info.wavName     = wavName;
    info.wavLevels   = wavLevels;
    info.domainFlag  = char(dom);

    % --------- 小波：生成 subbands ---------
    if wavLevels > 0

        [C,L] = wavedec(x, wavLevels, wavName);

        % band 按层 m=1..L，每层输出 A_m 与 D_m
        energiesA = zeros(1, wavLevels);
        energiesD = zeros(1, wavLevels);

        for m = 1:wavLevels
            bandA = wrcoef('a', C, L, wavName, m);
            bandD = wrcoef('d', C, L, wavName, m);

            bandNameA = sprintf('A%d', m);
            bandNameD = sprintf('D%d', m);

            % ---- A: time ----
            if useT
                [statsA_t, peakA_t] = basic_stats_and_peak(bandA);
                corrA = corr_safe(bandA, x);

                [feat, featNames, bandList, domList, statList] = ...
                    append_time_block(feat, featNames, bandList, domList, statList, ...
                    sensorTypeStr, bandNameA, statsA_t, peakA_t, corrA);
            end

            % ---- D: time ----
            if useT
                [statsD_t, peakD_t] = basic_stats_and_peak(bandD);
                corrD = corr_safe(bandD, x);

                [feat, featNames, bandList, domList, statList] = ...
                    append_time_block(feat, featNames, bandList, domList, statList, ...
                    sensorTypeStr, bandNameD, statsD_t, peakD_t, corrD);
            end

            % ---- A: freq ----
            if useF
                P1A = single_sided_amp(bandA, fs);
                [statsA_f, peakA_f] = basic_stats_and_peak(P1A);
                [scA, ssA] = spectral_centroid_spread(P1A, fs);

                % 记录能量
                energiesA(m) = sum(P1A.^2);

                [feat, featNames, bandList, domList, statList] = ...
                    append_freq_block(feat, featNames, bandList, domList, statList, ...
                    sensorTypeStr, bandNameA, statsA_f, peakA_f, scA, ssA);
            end

            % ---- D: freq ----
            if useF
                P1D = single_sided_amp(bandD, fs);
                [statsD_f, peakD_f] = basic_stats_and_peak(P1D);
                [scD, ssD] = spectral_centroid_spread(P1D, fs);

                energiesD(m) = sum(P1D.^2);

                [feat, featNames, bandList, domList, statList] = ...
                    append_freq_block(feat, featNames, bandList, domList, statList, ...
                    sensorTypeStr, bandNameD, statsD_f, peakD_f, scD, ssD);
            end
        end

        % --------- BER（仅当 useF 时有意义）---------
        if useF
            denom = sum(energiesA) + sum(energiesD);
            if denom <= 0
                berA = zeros(1, wavLevels);
                berD = zeros(1, wavLevels);
            else
                berA = energiesA / denom;
                berD = energiesD / denom;
            end

            % 先 A1..AL 再 D1..DL
            for m = 1:wavLevels
                bname = sprintf('A%d', m);
                [feat, featNames, bandList, domList, statList] = ...
                    append_scalar(feat, featNames, bandList, domList, statList, ...
                    sensorTypeStr, bname, 'F', 'BER', berA(m));
            end
            for m = 1:wavLevels
                bname = sprintf('D%d', m);
                [feat, featNames, bandList, domList, statList] = ...
                    append_scalar(feat, featNames, bandList, domList, statList, ...
                    sensorTypeStr, bname, 'F', 'BER', berD(m));
            end
        end

    else
        % ---------  wavLevels==0：直接对 raw 做特征 ---------
        bandName = 'raw';

        if useT
            [statsT, peakT] = basic_stats_and_peak(x);
            % raw 情况不加 corr（避免 corr(x,x)=1 ）
            [feat, featNames, bandList, domList, statList] = ...
                append_time_block_noCorr(feat, featNames, bandList, domList, statList, ...
                sensorTypeStr, bandName, statsT, peakT);
        end

        if useF
            P1 = single_sided_amp(x, fs);
            [statsF, peakF] = basic_stats_and_peak(P1);
            [sc, ss] = spectral_centroid_spread(P1, fs);

            [feat, featNames, bandList, domList, statList] = ...
                append_freq_block(feat, featNames, bandList, domList, statList, ...
                sensorTypeStr, bandName, statsF, peakF, sc, ss);
        end
    end

    % --------- 输出整理 ---------
    featRow = reshape(feat, 1, []);

    info.featNames = featNames;
    info.band      = bandList;
    info.domain    = domList;
    info.statName  = statList;

    % FFT 相关信息
    info.N              = N;
    info.LengthEff      = 2*(N-1);
    info.n_fft          = 2^nextpow2(info.LengthEff);

end

% ======================== Helper functions ========================

function [stats, peak_factor] = basic_stats_and_peak(x)
    x = double(x(:))';
    stats = zeros(1,12);

    stats(1)  = mean(x);
    stats(2)  = median(x);
    stats(3)  = mode(x);
    stats(4)  = range(x);
    stats(5)  = sum(abs(x - mean(x))) / length(x);             % mean abs deviation
    stats(6)  = std(x);
    stats(7)  = std(x) / mean(x);                              % coef of variation (may be Inf)
    stats(8)  = quantile(x,0.75) - quantile(x,0.25);           % IQR
    stats(9)  = skewness(x);
    stats(10) = kurtosis(x);
    stats(11) = sum(x.^2);                                     % energy
    stats(12) = info_entropy_256(x);                           % entropy (quantized)

    peak_value = max(x);
    rms_value  = sqrt(mean(x.^2));
    peak_factor = peak_value / rms_value;                      % may be Inf if rms=0
end

function H = info_entropy_256(signal)
    signal = double(signal(:));
    Nq = 256;

    smin = min(signal);
    smax = max(signal);
    if smax == smin
        H = 0;
        return;
    end

    q = round((signal - smin) / (smax - smin) * (Nq - 1)) + 1;  % 1..256
    uq = unique(q);

    p = histcounts(q, uq) / numel(q);
    H = -sum(p .* log2(p + eps));
end

function P1 = single_sided_amp(x, fs) 

    x = double(x(:))';
    N = numel(x);
    LengthEff = 2*(N-1);
    n_fft = 2^nextpow2(LengthEff);

    Y = fft(x, n_fft);
    P2 = abs(Y / LengthEff);

    P1 = P2(1:LengthEff/2+1);      % 长度 = N
    if numel(P1) > 2
        P1(2:end-1) = 2*P1(2:end-1);
    end
end

function [sc, ss] = spectral_centroid_spread(spec, fs)
    spec = double(spec(:))';
    n = numel(spec);
    denom = sum(spec);
    if denom <= 0
        sc = 0;
        ss = 0;
        return;
    end

    f = (0:n-1) * (fs / n);           
    w = spec / denom;

    sc = sum(f .* w);
    ss = sqrt(sum(((f - sc).^2) .* w));
end

function c = corr_safe(a, b)
    a = double(a(:));
    b = double(b(:));
    if numel(a) ~= numel(b) || numel(a) < 2
        c = NaN;
        return;
    end
    if std(a) == 0 || std(b) == 0
        c = NaN;
        return;
    end
    c = corr(a, b);
end

function [feat, featNames, bandList, domList, statList] = append_time_block( ...
    feat, featNames, bandList, domList, statList, sensorName, bandName, stats, peak_factor, corr_to_raw)

    % time: 12 stats + peak + corr = 14
    statNames12 = {'mean','median','mode','range','mad_mean','std','cv','iqr', ...
                   'skewness','kurtosis','energy','entropy'};

    % 12 stats
    for i = 1:12
        [feat, featNames, bandList, domList, statList] = append_scalar( ...
            feat, featNames, bandList, domList, statList, ...
            sensorName, bandName, 'T', statNames12{i}, stats(i));
    end
    % peak
    [feat, featNames, bandList, domList, statList] = append_scalar( ...
        feat, featNames, bandList, domList, statList, ...
        sensorName, bandName, 'T', 'peak_factor', peak_factor);

    % corr
    [feat, featNames, bandList, domList, statList] = append_scalar( ...
        feat, featNames, bandList, domList, statList, ...
        sensorName, bandName, 'T', 'corr_to_raw', corr_to_raw);
end

function [feat, featNames, bandList, domList, statList] = append_time_block_noCorr( ...
    feat, featNames, bandList, domList, statList, sensorName, bandName, stats, peak_factor)

    % time: 12 stats + peak = 13（raw 情况不加 corr）
    statNames12 = {'mean','median','mode','range','mad_mean','std','cv','iqr', ...
                   'skewness','kurtosis','energy','entropy'};

    for i = 1:12
        [feat, featNames, bandList, domList, statList] = append_scalar( ...
            feat, featNames, bandList, domList, statList, ...
            sensorName, bandName, 'T', statNames12{i}, stats(i));
    end
    [feat, featNames, bandList, domList, statList] = append_scalar( ...
        feat, featNames, bandList, domList, statList, ...
        sensorName, bandName, 'T', 'peak_factor', peak_factor);
end

function [feat, featNames, bandList, domList, statList] = append_freq_block( ...
    feat, featNames, bandList, domList, statList, sensorName, bandName, stats, peak_factor, sc, ss)

    % freq: 12 stats + peak + sc + ss = 15
    statNames12 = {'mean','median','mode','range','mad_mean','std','cv','iqr', ...
                   'skewness','kurtosis','energy','entropy'};

    for i = 1:12
        [feat, featNames, bandList, domList, statList] = append_scalar( ...
            feat, featNames, bandList, domList, statList, ...
            sensorName, bandName, 'F', statNames12{i}, stats(i));
    end
    [feat, featNames, bandList, domList, statList] = append_scalar( ...
        feat, featNames, bandList, domList, statList, ...
        sensorName, bandName, 'F', 'peak_factor', peak_factor);

    [feat, featNames, bandList, domList, statList] = append_scalar( ...
        feat, featNames, bandList, domList, statList, ...
        sensorName, bandName, 'F', 'spectral_centroid', sc);

    [feat, featNames, bandList, domList, statList] = append_scalar( ...
        feat, featNames, bandList, domList, statList, ...
        sensorName, bandName, 'F', 'spectral_spread', ss);
end

function [feat, featNames, bandList, domList, statList] = append_scalar( ...
    feat, featNames, bandList, domList, statList, sensorName, bandName, domChar, statName, value)

    feat(end+1) = value; 
    featNames{end+1} = sprintf('%s_%s_%s_%s', sensorName, bandName, domChar, statName); 
    bandList{end+1}  = bandName; 
    domList{end+1}   = domChar; 
    statList{end+1}  = statName; 
end
