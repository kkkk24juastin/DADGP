function [throughputMbps, ber, paprDb, energyPerBit, energyEfficiency] = evaluate_ofdm_point( ...
    x1, x2, x3, x4, modulationOrder, simulationSeed, simConfig)
% EVALUATE_OFDM_POINT
% 面向 MATLAB Engine 的轻量封装，用于调用 ofdm_link_quality_ex。
%
% 返回 Python Pareto 分析使用的四个工程目标：throughput、BER、PAPR 和
% energy efficiency。能耗模型与 analyze_real_simulation_pareto.py 及稳健分析
% 脚本保持一致。

if nargin < 5 || isempty(modulationOrder)
    modulationOrder = 16;
end
if nargin < 6 || isempty(simulationSeed)
    simulationSeed = 42;
end
if nargin < 7
    simConfig = [];
end

[metrics, ~] = ofdm_link_quality_ex( ...
    x1, ...
    x2, ...
    x3, ...
    x4, ...
    modulationOrder, ...
    simulationSeed, ...
    simConfig ...
);

throughputMbps = metrics.thr_mbps;
ber = metrics.ber;
paprDb = metrics.papr_dB;
[energyPerBit, energyEfficiency] = derive_energy_metrics(x1, throughputMbps);
end

function [energyPerBit, energyEfficiency] = derive_energy_metrics(ptxDbm, throughputMbps)
powerModelPbb = 0.2;
powerModelPrf = 0.8;
powerModelEtaPa = 0.35;
minThroughputMbps = 1e-6;

poutW = 10 .^ ((ptxDbm - 30.0) / 10.0);
totalPowerW = powerModelPbb + powerModelPrf + poutW / powerModelEtaPa;
bitrateBps = max(throughputMbps, minThroughputMbps) * 1e6;

energyPerBit = totalPowerW ./ bitrateBps;
energyEfficiency = 1 ./ energyPerBit;
end
