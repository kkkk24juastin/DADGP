% GENERATE_OFDM_TRAIN_VAL_DATASET
% 基于当前仓库中的 OFDM 仿真生成训练集和验证集。
%
% 输入参数定义:
%   x1 = Ptx_dBm
%   x2 = d_m
%   x3 = tauRMS_ns
%   x4 = cfo_ppm
% 输出指标来自 ofdm_link_quality_ex.m:
%   y1 = throughput_mbps
%   y2 = ber
%   y3 = papr_db
%
% 输出文件:
%   data/train.xlsx
%   data/val.xlsx
%
% 文件格式与当前 Python 训练代码保持一致:
%   前 4 列为输入 x1-x4，后 3 列为输出 y1-y3。
%
% 运行方式:
%   >> generate_ofdm_train_val_dataset

clearvars -except ...
    nTrainOverride ...
    nValOverride ...
    modulationOrderOverride ...
    simulationSeedOverride ...
    outputDataDirOverride;
clc;

%% -------------------- User Config --------------------
nTrain = 400;
nVal = 400;
modulationOrder = 16;
xTrainSeed = 42;
xValSeed = 24;
simulationSeed = 42;

if exist('nTrainOverride', 'var') && ~isempty(nTrainOverride)
    nTrain = nTrainOverride;
end
if exist('nValOverride', 'var') && ~isempty(nValOverride)
    nVal = nValOverride;
end
if exist('modulationOrderOverride', 'var') && ~isempty(modulationOrderOverride)
    modulationOrder = modulationOrderOverride;
end
if exist('simulationSeedOverride', 'var') && ~isempty(simulationSeedOverride)
    simulationSeed = simulationSeedOverride;
end

% 与 Python 端 MOO 搜索边界保持一致:
% [Ptx_dBm, d_m, tauRMS_ns, cfo_ppm]
lowerBounds = [5.0, 50.0, 50.0, 0.0];
upperBounds = [30.0, 500.0, 500.0, 10.0];

%% -------------------- Path Setup ---------------------
scriptPath = mfilename('fullpath');
if isempty(scriptPath)
    projectDir = pwd;
else
    projectDir = fileparts(scriptPath);
end

if exist('outputDataDirOverride', 'var') && ~isempty(outputDataDirOverride)
    dataDir = outputDataDirOverride;
else
    dataDir = fullfile(projectDir, 'data');
end
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

trainFile = fullfile(dataDir, 'train.xlsx');
valFile = fullfile(dataDir, 'val.xlsx');

%% -------------------- Sample Inputs ------------------
rng(xTrainSeed, 'twister');
XTrain = latin_hypercube_uniform(nTrain, lowerBounds, upperBounds);
rng(xValSeed, 'twister');
XVal = latin_hypercube_uniform(nVal, lowerBounds, upperBounds);

%% -------------------- Run Simulations ----------------
YTrain = simulate_dataset( ...
    XTrain, ...
    modulationOrder, ...
    simulationSeed, ...
    'train' ...
);
YVal = simulate_dataset( ...
    XVal, ...
    modulationOrder, ...
    simulationSeed, ...
    'val' ...
);

%% -------------------- Save Splits --------------------
trainTable = array2table( ...
    [XTrain, YTrain], ...
    'VariableNames', {'x1', 'x2', 'x3', 'x4', 'y1', 'y2', 'y3'} ...
);

valTable = array2table( ...
    [XVal, YVal], ...
    'VariableNames', {'x1', 'x2', 'x3', 'x4', 'y1', 'y2', 'y3'} ...
);

writetable(trainTable, trainFile);
writetable(valTable, valFile);

fprintf('\nSaved training set to: %s\n', trainFile);
fprintf('Saved validation set to: %s\n', valFile);

print_dataset_summary('train', trainTable);
print_dataset_summary('val', valTable);
print_cross_split_summary(trainTable, valTable);

%% -------------------- Local Functions ----------------
function X = latin_hypercube_uniform(nSamples, lowerBounds, upperBounds)
% 使用不依赖 Statistics Toolbox 的简易 LHS 采样。
% 依赖外部已经设定好的 RNG 状态，这样 train/val 可以在同一个 seed 下独立采样。
    lowerBounds = reshape(lowerBounds, 1, []);
    upperBounds = reshape(upperBounds, 1, []);
    nDims = numel(lowerBounds);

    if numel(upperBounds) ~= nDims
        error('lowerBounds and upperBounds must have the same length.');
    end

    if any(upperBounds <= lowerBounds)
        error('Each upper bound must be greater than its lower bound.');
    end

    unitSamples = zeros(nSamples, nDims);
    for d = 1:nDims
        intervalStarts = (0:(nSamples - 1))' / nSamples;
        intervalJitter = rand(nSamples, 1) / nSamples;
        stratified = intervalStarts + intervalJitter;
        unitSamples(:, d) = stratified(randperm(nSamples));
    end

    span = upperBounds - lowerBounds;
    X = repmat(lowerBounds, nSamples, 1) + unitSamples .* repmat(span, nSamples, 1);
end

function Y = simulate_dataset(X, modulationOrder, simulationSeed, splitName)
    nSamples = size(X, 1);
    Y = zeros(nSamples, 3);

    fprintf('Generating %s set with %d deterministic simulation samples...\n', splitName, nSamples);

    for i = 1:nSamples
        metrics = ofdm_link_quality_ex( ...
            X(i, 1), ...
            X(i, 2), ...
            X(i, 3), ...
            X(i, 4), ...
            modulationOrder, ...
            simulationSeed ...
        );
        Y(i, :) = [metrics.thr_mbps, metrics.ber, metrics.papr_dB];

        if mod(i, 25) == 0 || i == nSamples
            fprintf('  %s completed %d / %d\n', splitName, i, nSamples);
        end
    end
end

function print_dataset_summary(splitName, tbl)
    X = tbl{:, 1:4};
    Y = tbl{:, 5:7};
    yMean = mean(Y, 1);
    yStd = std(Y, 0, 1);

    fprintf('\n[%s] size: %d samples\n', splitName, size(tbl, 1));
    fprintf('  X min: [%g, %g, %g, %g]\n', min(X, [], 1));
    fprintf('  X max: [%g, %g, %g, %g]\n', max(X, [], 1));
    for taskIdx = 1:size(Y, 2)
        fprintf( ...
            '  y%d min/max/mean/std: %g / %g / %g / %g\n', ...
            taskIdx, ...
            min(Y(:, taskIdx)), ...
            max(Y(:, taskIdx)), ...
            yMean(taskIdx), ...
            yStd(taskIdx) ...
        );
    end

    if size(Y, 1) > 1
        corrY12 = corr(Y(:, 1), Y(:, 2));
    else
        corrY12 = NaN;
    end
    fprintf('  corr(y1, y2): %g\n', corrY12);
    fprintf('  std(y3): %g\n', yStd(3));
end

function print_cross_split_summary(trainTable, valTable)
    trainY = trainTable{:, 5:7};
    valY = valTable{:, 5:7};
    meanDiff = mean(valY, 1) - mean(trainY, 1);

    fprintf('\n[train-val] output mean diff (val - train): [%g, %g, %g]\n', meanDiff);
end
