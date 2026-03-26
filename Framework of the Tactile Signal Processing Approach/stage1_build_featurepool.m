function [featurepool, output_class, meta] = stage1_build_featurepool(dataset3, fs, num)
% STAGE 1: 生成 386 维 featurepool
% Inputs
%   dataset3 : 36x60 cell，每格 4x2001 double（pvdf1,pvdf2,sg1,sg2）
%   fs       : 采样率
%   num      : 结构体，至少包含 num.shape, num.trial, num.etr
%              e.g. num.shape=36; num.trial=60; num.etr=1;
% Outputs
%   featurepool  : (num.trial*num.etr*num.shape) x 386
%   output_class : 长度与 featurepool 行数一致
%   meta         : 辅助信息（n_fft, wavf, N_wavlayer 等）
%

len_signal = 2;
dataset1   = dataset3;
wavf       = 'rbio2.2';
N_wavlayer = 3;
Num_point  = 2*len_signal*fs;

% pvdf1, pvdf2: time & freq domain
% sg1, sg2: time-domain
num_fea_cloumn = 58;
num_fea_row    = N_wavlayer*2;
num.feature    = num_fea_row*num_fea_cloumn + num_fea_row*2 + 13*2; % =386
featurepool    = zeros(num.trial*num.etr*num.shape, num.feature);

readme = {'featurepool'; ...
    '1-14:   PVDF1 A1 time-domain (14).'; ...
    '15-28:  PVDF1 D1 time-domain (14).'; ...
    '29-43:  PVDF1 A1 frequency-domain (15).'; ...
    '44-58:  PVDF1 D1 frequency-domain (15).'; ...
    '59-72:  PVDF2 A1 time-domain (14).'; ...
    '73-86:  PVDF2 D1 time-domain (14).'; ...
    '87-101: PVDF2 A1 frequency-domain (15).'; ...
    '102-116:PVDF2 D1 frequency-domain (15).'; ...
    '117-130:PVDF1 A2 time-domain (14).'; ...
    '131-144:PVDF1 D2 time-domain (14).'; ...
    '145-159:PVDF1 A2 frequency-domain (15).'; ...
    '160-174:PVDF1 D2 frequency-domain (15).'; ...
    '175-188:PVDF2 A2 time-domain (14).'; ...
    '189-202:PVDF2 D2 time-domain (14).'; ...
    '203-217:PVDF2 A2 frequency-domain (15).'; ...
    '218-232:PVDF2 D2 frequency-domain (15).'; ...
    '233-246:PVDF1 A3 time-domain (14).'; ...
    '247-260:PVDF1 D3 time-domain (14).'; ...
    '261-275:PVDF1 A3 frequency-domain (15).'; ...
    '276-290:PVDF1 D3 frequency-domain (15).'; ...
    '291-304:PVDF2 A3 time-domain (14).'; ...
    '305-318:PVDF2 D3 time-domain (14).'; ...
    '319-333:PVDF2 A3 frequency-domain (15).'; ...
    '334-348:PVDF2 D3 frequency-domain (15).'; ...
    '349-354:PVDF1 frequency-domain BER for A1,D1,A2,D2,A3,D3 (6). [from cols 39,54,155,170,271,286 normalized]'; ...
    '355-360:PVDF2 frequency-domain BER for A1,D1,A2,D2,A3,D3 (6). [from cols 97,112,213,228,329,344 normalized]'; ...
    '361-373:SG1 time-domain features (13).'; ...
    '374-386:SG2 time-domain features (13).'; ...
    'Notes: num_fea_cloumn=58; N_wavlayer=3; each PVDF block = 14(A) + 14(D) + 15(A-freq) + 15(D-freq).'};



% ------------------ 原始 FFT 参数 ------------------
T      = 1/fs;
Length = Num_point;
t      = (0:Length-1)*T;
n_fft  = 2^nextpow2(Length);

Q         = 30;
harmonics = 150:50:450; % 50Hz 的整数倍
num.filter   = length(harmonics);
filter_ab    = cell(num.filter,2);
dataset_shifted = cell(size(dataset1));

% ------------------ 陷波 + 复制 ------------------
for i = 1:num.shape
    for j = 1:num.etr*num.trial
        if isempty(dataset1{i,j}), continue; end 
        temp1 = dataset1{i,j}(1,:);
        temp2 = dataset1{i,j}(2,:);
        for k = 1:num.filter
            BW = harmonics(k)/Q;
            f0 = harmonics(k);
            [filter_ab{k,1},filter_ab{k,2}] = iirnotch(f0/(fs/2), BW/(fs/2));
            temp1 = filtfilt(filter_ab{k,1},filter_ab{k,2}, temp1);
            temp2 = filtfilt(filter_ab{k,1},filter_ab{k,2}, temp2);
        end
        dataset_shifted{i,j}(1,:) = temp1;
        dataset_shifted{i,j}(2,:) = temp2;
        dataset_shifted{i,j}(3,:) = dataset1{i,j}(3,:);
        dataset_shifted{i,j}(4,:) = dataset1{i,j}(4,:);
    end
end

% ------------------ 小波分解 + 特征提取 ------------------
for i = 1:num.shape
    for j = 1:num.trial
        for k = 1:num.etr
            if isempty(dataset_shifted{i,(j-1)*num.etr+k}), continue; end % 健壮性
            [row1,col1] = wavedec(dataset_shifted{i,(j-1)*num.etr+k}(1,:), N_wavlayer, wavf);
            [row2,col2] = wavedec(dataset_shifted{i,(j-1)*num.etr+k}(2,:), N_wavlayer, wavf);
            for m = 1:N_wavlayer
                sensordata.a1{i}{j,k}(m,:) = wrcoef('a', row1,col1,wavf,m);  % pvdf1
                sensordata.d1{i}{j,k}(m,:) = wrcoef('d', row1,col1,wavf,m);
                sensordata.a2{i}{j,k}(m,:) = wrcoef('a', row2,col2,wavf,m);  % pvdf2
                sensordata.d2{i}{j,k}(m,:) = wrcoef('d', row2,col2,wavf,m);

                sensordata.af1{i}{j,k}(m,:) = fft(sensordata.a1{i}{j,k}(m,:), n_fft);
                P2.af1 = abs(sensordata.af1{i}{j,k}(m,:)/Length);
                P1.af1 = P2.af1(1:Length/2+1);
                P1.af1(2:end-1) = 2*P1.af1(2:end-1);

                sensordata.df1{i}{j,k}(m,:) = fft(sensordata.d1{i}{j,k}(m,:), n_fft);
                P2.df1 = abs(sensordata.df1{i}{j,k}(m,:)/Length);
                P1.df1 = P2.df1(1:Length/2+1);
                P1.df1(2:end-1) = 2*P1.df1(2:end-1);

                sensordata.af2{i}{j,k}(m,:) = fft(sensordata.a2{i}{j,k}(m,:), n_fft);
                P2.af2 = abs(sensordata.af2{i}{j,k}(m,:)/Length);
                P1.af2 = P2.af2(1:Length/2+1);
                P1.af2(2:end-1) = 2*P1.af2(2:end-1);

                sensordata.df2{i}{j,k}(m,:) = fft(sensordata.d2{i}{j,k}(m,:), n_fft);
                P2.df2 = abs(sensordata.df2{i}{j,k}(m,:)/Length);
                P1.df2 = P2.df2(1:Length/2+1);
                P1.df2(2:end-1) = 2*P1.df2(2:end-1);

                %%% PVDF1
                id_row = i + num.shape*(num.etr*(j-1)+(k-1));
                [stats_at1,peak_factor_at1,~,~] = Signal_features(sensordata.a1{i}{j,k}(m,:));
                corr_at1 = corr(sensordata.a1{i}{j,k}(m,:)', dataset_shifted{i,(j-1)*num.etr+k}(1,:)');
                [stats_dt1,peak_factor_dt1,~,~] = Signal_features(sensordata.d1{i}{j,k}(m,:));
                corr_dt1 = corr(sensordata.d1{i}{j,k}(m,:)', dataset_shifted{i,(j-1)*num.etr+k}(1,:)');

                [stats_af1,peak_factor_af1,sc_af1,ss_af1] = Signal_features(P1.af1);
                [stats_df1,peak_factor_df1,sc_df1,ss_df1] = Signal_features(P1.df1);

                featurepool(id_row, num_fea_cloumn*2*(m-1)+(1:num_fea_cloumn)) = ...
                    [stats_at1,peak_factor_at1,corr_at1, ...
                     stats_dt1,peak_factor_dt1,corr_dt1, ...
                     stats_af1,peak_factor_af1,sc_af1,ss_af1, ...
                     stats_df1,peak_factor_df1,sc_df1,ss_df1];

                %%% PVDF2
                [stats_at2,peak_factor_at2,~,~] = Signal_features(sensordata.a2{i}{j,k}(m,:));
                corr_at2 = corr(sensordata.a2{i}{j,k}(m,:)', dataset_shifted{i,(j-1)*num.etr+k}(2,:)');
                [stats_dt2,peak_factor_dt2,~,~] = Signal_features(sensordata.d2{i}{j,k}(m,:));
                corr_dt2 = corr(sensordata.d2{i}{j,k}(m,:)', dataset_shifted{i,(j-1)*num.etr+k}(2,:)');

                [stats_af2,peak_factor_af2,sc_af2,ss_af2] = Signal_features(P1.af2);
                [stats_df2,peak_factor_df2,sc_df2,ss_df2] = Signal_features(P1.df2);

                featurepool(id_row, num_fea_cloumn*(2*m-1)+(1:num_fea_cloumn)) = ...
                    [stats_at2,peak_factor_at2,corr_at2, ...
                     stats_dt2,peak_factor_dt2,corr_dt2, ...
                     stats_af2,peak_factor_af2,sc_af2,ss_af2, ...
                     stats_df2,peak_factor_df2,sc_df2,ss_df2];
            end

            featurepool(id_row, num_fea_cloumn*2*m+(1:num_fea_row)) = ...
                featurepool(id_row, [num_fea_cloumn*(0:2:2*(N_wavlayer-1))+39, num_fea_cloumn*(0:2:2*(N_wavlayer-1))+54]) ...
               /sum(featurepool(id_row, [num_fea_cloumn*(0:2:2*(N_wavlayer-1))+39, num_fea_cloumn*(0:2:2*(N_wavlayer-1))+54]));

            featurepool(id_row, num_fea_cloumn*2*m+((num_fea_row+1):2*num_fea_row)) = ...
                featurepool(id_row, [num_fea_cloumn*(1:2:(2*N_wavlayer-1))+39, num_fea_cloumn*(1:2:(2*N_wavlayer-1))+54]) ...
               /sum(featurepool(id_row, [num_fea_cloumn*(1:2:(2*N_wavlayer-1))+39, num_fea_cloumn*(1:2:(2*N_wavlayer-1))+54]));

            %%% SG1
            [stats_sg1,peak_factor_sg1,~,~] = Signal_features(dataset_shifted{i,(j-1)*num.etr+k}(3,:));
            featurepool(id_row, num_fea_cloumn*2*m+num_fea_row*2+(1:13)) = [stats_sg1,peak_factor_sg1];

            %%% SG2
            [stats_sg2,peak_factor_sg2,~,~] = Signal_features(dataset_shifted{i,(j-1)*num.etr+k}(4,:));
            featurepool(id_row, num_fea_cloumn*2*m+num_fea_row*2+(14:26)) = [stats_sg2,peak_factor_sg2];
        end
    end
end


out_trial    = ones(num.group*num.trial*num.etr,1) * (1:num.shape);
output_class = reshape(out_trial', [], 1);

% ------------------ meta 信息 ------------------
meta.readme      = readme;
meta.wavf        = wavf;
meta.N_wavlayer  = N_wavlayer;
meta.Num_point   = Num_point;
meta.n_fft       = n_fft;
meta.harmonics   = harmonics;
meta.Q           = Q;
meta.fs          = fs;
meta.num         = num;


assert(size(featurepool,2)==386, 'featurepool 列数不是 386！');

if any(~isfinite(featurepool(:)))
    warning('featurepool 中存在 NaN/Inf，请检查输入数据是否有空 cell 或异常。');
end

end 


function [stats, peak_factor, spectral_centroid, spectral_spread] = Signal_features(x)
    stats(1)=mean(x);
    stats(2)=median(x);
    stats(3)=mode(x);
    stats(4)=range(x);
    stats(5)=sum(abs(x-mean(x)))/length(x);
    stats(6)=std(x);
    stats(7)=std(x)/mean(x);
    stats(8)=quantile(x,0.75)- quantile(x,0.25);
    stats(9)=skewness(x);
    stats(10)=kurtosis(x);
    stats(11)=sum(x.^2);

    signal=x;
    N = 256;
    quantized_signal = round((signal - min(signal)) / (max(signal) - min(signal)) * (N - 1)) + 1;
    unique_values = unique(quantized_signal);
    probabilities = histcounts(quantized_signal, unique_values) / numel(quantized_signal);
    info_entropy = -sum(probabilities .* log2(probabilities + eps));
    stats(12)=info_entropy;

    peak_value = max(signal);
    rms_value  = sqrt(mean(signal.^2));
    peak_factor = peak_value / rms_value;

    fs=1000;
    n_fft = length(x);
    freq_vector = (0:n_fft-1) * (fs / n_fft);
    spectral_centroid = sum(freq_vector .* (x / sum(x)));
    deviations = (freq_vector - spectral_centroid) .^ 2;
    spectral_spread = sqrt(sum(deviations .* (x / sum(x))));
end
