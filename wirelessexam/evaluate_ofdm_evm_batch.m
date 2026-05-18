function metricsMatrix = evaluate_ofdm_evm_batch( ...
    x1, x2, x3, x4, modulationOrder, simulationSeed, simConfig)
% EVALUATE_OFDM_EVM_BATCH
% 面向 Python MATLAB Engine 调用的批量评估辅助函数。
%
% 每个候选点返回一行:
%   [evm_rms, ber, papr_dB, throughput_mbps, snr_dB]

if nargin < 5 || isempty(modulationOrder)
    modulationOrder = 16;
end
if nargin < 6 || isempty(simulationSeed)
    simulationSeed = 42;
end
if nargin < 7
    simConfig = [];
end

x1 = x1(:);
x2 = x2(:);
x3 = x3(:);
x4 = x4(:);

numRows = numel(x1);
if any([numel(x2), numel(x3), numel(x4)] ~= numRows)
    error('x1, x2, x3, and x4 must have the same number of rows.');
end

metricsMatrix = zeros(numRows, 5);

for rowIdx = 1:numRows
    [metrics, ~] = ofdm_link_quality_ex( ...
        x1(rowIdx), ...
        x2(rowIdx), ...
        x3(rowIdx), ...
        x4(rowIdx), ...
        modulationOrder, ...
        simulationSeed, ...
        simConfig ...
    );

    metricsMatrix(rowIdx, :) = [ ...
        metrics.evm_rms, ...
        metrics.ber, ...
        metrics.papr_dB, ...
        metrics.thr_mbps, ...
        metrics.snr_dB ...
    ];

    if mod(rowIdx, 25) == 0 || rowIdx == numRows
        fprintf('  EVM batch completed %d / %d\n', rowIdx, numRows);
    end
end
end
