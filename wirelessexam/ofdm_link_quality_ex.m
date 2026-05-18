
function [metrics, aux] = ofdm_link_quality_ex(Ptx_dBm, d_m, tauRMS_ns, cfo_ppm, M, rngseed, simConfig)
% 扩展版 OFDM 链路：支持 M-QAM，并返回多种链路质量指标。
% 输入:
%   Ptx_dBm, d_m, tauRMS_ns, cfo_ppm: 含义与原脚本一致
%   M: 调制阶数（例如 4、16、64）-> QPSK/16QAM/64QAM
%   rngseed: 当前随机实现的随机种子
%   simConfig: 可选结构体，用于 CFR / goodput 整形参数
% 输出:
%   metrics: 结构体字段包括
%       .ber           : 未编码误码率
%       .thr_mbps      : 有效吞吐量 / goodput (Mbps)
%       .snr_dB        : SNR (dB)
%       .evm_rms       : RMS EVM（线性值）
%       .papr_dB       : CFR 处理后 OFDM 波形的 PAPR (dB)
%   aux: 用于绘图 / 诊断的辅助数组
%       .rxDataSymEQ   : 均衡后的数据符号
%       .txDataSym     : 发送数据符号
%       .tx_time       : CFR 处理后的含 CP 时域 OFDM 向量
%       .fs            : 采样率
%       .freq_axis     : PSD 使用的归一化频率轴
%       .used_idx      : 已用子载波索引
%       .Y_used        : 接收频域符号
%       .quality_score : 平滑链路质量得分
%       .tx_stress     : 平滑发射端压力得分
%       .clip_ratio_dB : 自适应 CFR 削波比
%       .raw_rate_mbps : goodput 整形前的未编码 OFDM 载荷速率
%
if nargin < 6 || isempty(rngseed)
    rngseed = 42;
end
if nargin < 7
    simConfig = [];
end

simConfig = normalize_sim_config(simConfig);
rng(rngseed, 'twister');

%% ---------- System/OFDM ----------
fc      = 3.5e9;
Nfft    = 256;
Ncp     = 64;
Delta_f = 15e3;
Fs      = Nfft * Delta_f;
Nc_use  = 200;
Nsym    = 120;   % a bit more to stabilize EVM / PAPR
NF_dB   = 5;
n_PL    = 3.0;
d0      = 1;

mid = Nfft / 2 + 1;
left  = mid - ceil(Nc_use / 2);
right = mid + floor(Nc_use / 2) - 1;
sc_mask = false(1, Nfft);
sc_mask(left:right) = true;
sc_mask(mid) = false;
used_idx = find(sc_mask);
Nc_use = numel(used_idx);

isPilot = false(Nc_use, 1);
isPilot(1:8:end) = true;
Np = sum(isPilot);
Nd = Nc_use - Np;

% QAM bits per symbol
kbits = log2(M);

%% ---------- Link budget ----------
c = 3e8;
PL0_dB = 20 * log10(4 * pi * fc / c);
PL_dB  = PL0_dB + 10 * n_PL * log10(max(d_m / d0, 1));
BW_noise = Fs;
N0_dBm = -174 + 10 * log10(BW_noise) + NF_dB;
SNRdB  = Ptx_dBm - PL_dB - N0_dBm;

u_ptx = clamp_value((Ptx_dBm - 5.0) / 25.0, 0.0, 1.0);
u_tau = clamp_value((tauRMS_ns - 50.0) / 450.0, 0.0, 1.0);
u_cfo = clamp_value(cfo_ppm / 10.0, 0.0, 1.0);
snr_soft = 1.0 / (1.0 + exp(-(SNRdB - 12.0) / 4.0));
quality_score = clamp_value( ...
    0.10 + 0.75 * snr_soft - 0.25 * u_tau - 0.30 * u_cfo, ...
    0.05, ...
    0.95 ...
);
tx_stress = clamp_value(0.60 * u_ptx + 0.25 * u_cfo + 0.15 * u_tau, 0.0, 1.0);

%% ---------- Channel from tauRMS ----------
tau_rms = max(tauRMS_ns, 1) * 1e-9;
Ts = 1 / Fs;
L = 14;
delays = (0:L - 1) * Ts;
pdp = exp(-delays / max(tau_rms, Ts));
pdp = pdp / sum(pdp);
h = (randn(1, L) + 1j * randn(1, L)) ./ sqrt(2) .* sqrt(pdp);
maxDelay = L - 1;

%% ---------- Tx grid (pilots + data), M-QAM ----------
Nb_total = Nd * kbits * Nsym;
txBits = randi([0 1], Nb_total, 1);

mapTable = qammod(0:M - 1, M, 'gray', 'UnitAveragePower', true);
bin = @(n) dec2bin(n, kbits) - '0';
sym2bits = zeros(M, kbits);
for ii = 1:M
    sym2bits(ii, :) = bin(ii - 1);
end

X = complex(zeros(Nfft, Nsym));
txDataSym = complex(zeros(Nd, Nsym));
pilotSym = (1 + 1j) / sqrt(2);
kb = 1;
for n = 1:Nsym
    used_syms = complex(zeros(Nc_use, 1));
    used_syms(isPilot) = pilotSym;
    data_idx = find(~isPilot);
    for m = 1:Nd
        bits = txBits(kb:kb + kbits - 1).';
        [~, idx] = ismember(bits, sym2bits, 'rows');
        idx = idx - 1;
        used_syms(data_idx(m)) = mapTable(idx + 1);
        txDataSym(m, n) = used_syms(data_idx(m));
        kb = kb + kbits;
    end
    X(used_idx, n) = used_syms;
end

% OFDM modulation
x_noCP = ifft(X, Nfft, 1) * sqrt(Nfft);
x_cp = [x_noCP(end - Ncp + 1:end, :); x_noCP];
tx = x_cp(:);

%% ---------- Adaptive CFR ----------
clip_ratio_span = simConfig.clipRatioMaxDb - simConfig.clipRatioMinDb;
clip_ratio_dB = simConfig.clipRatioMaxDb - clip_ratio_span * tx_stress;
tx_rms = sqrt(mean(abs(tx) .^ 2));
A = max(tx_rms * 10 ^ (clip_ratio_dB / 20), 1e-12);
p = simConfig.paprSoftLimiterP;
tx_cfr = tx ./ (1 + (abs(tx) / A) .^ (2 * p)) .^ (1 / (2 * p));

papr_dB = 10 * log10(max(abs(tx_cfr) .^ 2) / mean(abs(tx_cfr) .^ 2));

%% ---------- Channel + CFO + AWGN ----------
rx_chan = conv(tx_cfr, h(:));
cfo_Hz = (cfo_ppm * 1e-6) * fc;
eps_cfo = cfo_Hz / Fs;
n = (0:length(rx_chan) - 1).';
rx_cfo = rx_chan .* exp(1j * 2 * pi * eps_cfo * n);

Es = mean(abs(tx_cfr) .^ 2);
N0 = Es * 10 ^ (-SNRdB / 10);
noise = sqrt(N0 / 2) * (randn(size(rx_cfo)) + 1j * randn(size(rx_cfo)));
rx = rx_cfo + noise;

rx = rx(1 + maxDelay:end);
rx = rx(1:floor(length(rx) / (Nfft + Ncp)) * (Nfft + Ncp));

%% ---------- Rx ----------
rxMat = reshape(rx, Nfft + Ncp, []);
rxNoCP = rxMat(Ncp + 1:end, :);
Y = fft(rxNoCP, Nfft, 1) / sqrt(Nfft);
Y_used = Y(used_idx, :);

H_est = complex(zeros(Nc_use, size(Y_used, 2)));
for n = 1:size(Y_used, 2)
    Yu = Y_used(:, n);
    Hpil = Yu(isPilot) / pilotSym;
    xp = find(isPilot);
    xi = (1:Nc_use).';
    H_est(:, n) = interp1(xp, Hpil, xi, 'linear', 'extrap');
end

EQ = Y_used ./ H_est;
dataEQ = EQ(~isPilot, :);
rxDataSym = dataEQ(:);
txDataSymVec = txDataSym(:);

rxBits = zeros(length(txBits), 1);
for i = 1:length(rxDataSym)
    [~, idxmin] = min(abs(rxDataSym(i) - mapTable));
    rxBits((i - 1) * kbits + 1:i * kbits) = sym2bits(idxmin, :).';
end

%% ---------- Metrics ----------
ber = mean(rxBits ~= txBits(1:length(rxBits)));

payload_bits_per_sym = Nd * kbits;
T_sym = (Nfft + Ncp) / Fs;
raw_rate_mbps = payload_bits_per_sym / T_sym / 1e6;

rate_scale_span = simConfig.throughputRateScaleMax - simConfig.throughputRateScaleMin;
rate_scale = simConfig.throughputRateScaleMin + rate_scale_span * quality_score;

overhead_scale_span = simConfig.throughputOverheadMax - simConfig.throughputOverheadMin;
overhead_scale = simConfig.throughputOverheadMax - overhead_scale_span * tx_stress;

error_scale = max(1e-6, 1 - ber) ^ simConfig.throughputBerExponent;
thr_mbps = raw_rate_mbps * rate_scale * overhead_scale * error_scale;

err = rxDataSym - txDataSymVec;
evm_rms = sqrt(mean(abs(err) .^ 2) / mean(abs(txDataSymVec) .^ 2));

Npsd = 8192;
TXseg = tx_cfr;
Xf = fftshift(fft(TXseg, Npsd)); %#ok<NASGU>
freq_axis = linspace(-0.5, 0.5, Npsd);

metrics = struct( ...
    'ber', ber, ...
    'thr_mbps', thr_mbps, ...
    'snr_dB', SNRdB, ...
    'evm_rms', evm_rms, ...
    'papr_dB', papr_dB ...
);
aux = struct( ...
    'rxDataSymEQ', rxDataSym, ...
    'txDataSym', txDataSymVec, ...
    'tx_time', tx_cfr, ...
    'fs', Fs, ...
    'freq_axis', freq_axis, ...
    'used_idx', used_idx, ...
    'Y_used', Y_used, ...
    'quality_score', quality_score, ...
    'tx_stress', tx_stress, ...
    'clip_ratio_dB', clip_ratio_dB, ...
    'raw_rate_mbps', raw_rate_mbps ...
);
end

function simConfig = normalize_sim_config(simConfig)
    defaults = default_sim_config();
    simConfig = merge_struct_fields(defaults, simConfig);
end

function defaults = default_sim_config()
    defaults = struct( ...
        'paprSoftLimiterP', 3, ...
        'clipRatioMinDb', 6.5, ...
        'clipRatioMaxDb', 9.5, ...
        'throughputRateScaleMin', 0.70, ...
        'throughputRateScaleMax', 1.00, ...
        'throughputOverheadMin', 0.84, ...
        'throughputOverheadMax', 0.92, ...
        'throughputBerExponent', 0.60 ...
    );
end

function merged = merge_struct_fields(defaults, overrides)
    merged = defaults;
    if isempty(overrides)
        return;
    end

    fieldNames = fieldnames(defaults);
    for idx = 1:numel(fieldNames)
        fieldName = fieldNames{idx};
        if isfield(overrides, fieldName) && ~isempty(overrides.(fieldName))
            merged.(fieldName) = overrides.(fieldName);
        end
    end
end

function value = clamp_value(value, lowerBound, upperBound)
    value = min(max(value, lowerBound), upperBound);
end
