% ROBUST_SELECT_OFDM_EXISTING_NOISE
% 仅使用现有随机仿真器，对 OFDM Pareto 候选点进行稳健仿真。
%
% 第一阶段保持不变：`moo/<method>.xlsx` 保存代理模型辅助 NSGA-II 得到的
% Pareto 候选点。
% 本脚本只针对每个候选点，在多个 rngseed 下重复运行现有 OFDM 仿真器中的
% 随机因素（比特序列、信道抽头、AWGN），并导出经验均值/方差列。稳健损失
% 计算、Pareto 筛选、稳健最优点选择和排序由 Python 分析脚本负责。
%
% 直接修改下方用户配置区域。
% `SELECTED_METHODS = {}` 表示扫描 INPUT_MOO_DIR 下的全部方法文件。
% `SELECTED_METHODS = {'dadgp'}` 表示只更新一个方法。

clearvars;
clc;

%% -------------------- 路径设置 ---------------------
scriptPath = mfilename('fullpath');
if isempty(scriptPath)
    projectDir = pwd;
else
    projectDir = fileparts(scriptPath);
end

%% -------------------- 用户配置 --------------------
SELECTED_METHODS = {'bo_qparego', 'bo_qehvi', 'bo_qnehvi'};
INPUT_MOO_DIR = fullfile(projectDir, 'moo');
OUTPUT_RESULT_DIR = fullfile(projectDir, 'result');
MODULATION_ORDER = 16;
SIMULATION_SEEDS = 42:57;
SIM_CONFIG = [];

INPUT_SHEET_NAME = 'results';
ALL_CANDIDATES_SHEET_NAME = 'all_candidates';
METADATA_SHEET_NAME = 'metadata';

%% -------------------- 配置解析 -----------------
selectedMethods = normalize_method_list(SELECTED_METHODS);
inputMooDir = char(string(INPUT_MOO_DIR));
outputResultDir = char(string(OUTPUT_RESULT_DIR));
modulationOrder = MODULATION_ORDER;
simulationSeeds = reshape(SIMULATION_SEEDS, 1, []);
simConfig = SIM_CONFIG;

if isempty(simulationSeeds)
    error('SIMULATION_SEEDS must not be empty.');
end
if ~exist(outputResultDir, 'dir')
    mkdir(outputResultDir);
end

inputFiles = discover_input_files(inputMooDir, selectedMethods);
if isempty(inputFiles)
    error('No available MOO input files were found.');
end

fprintf('Input MOO dir: %s\n', inputMooDir);
fprintf('Output result dir: %s\n', outputResultDir);
fprintf('Modulation order: %d\n', modulationOrder);
fprintf('Simulation seeds: %s\n', mat2str(simulationSeeds));
fprintf('Noise source: existing OFDM RNG realizations (bits, channel, AWGN), fixed NF=5 dB.\n');
if isempty(selectedMethods)
    fprintf('Selected methods: <all available>\n');
else
    fprintf('Selected methods: %s\n', strjoin(selectedMethods, ', '));
end

%% -------------------- 输入加载 --------------------
inputSpecs = load_input_specs(inputFiles, INPUT_SHEET_NAME);
if isempty(inputSpecs)
    error('No processable Pareto candidates were found in the input files.');
end

%% -------------------- 稳健仿真 ---------------
totalCandidates = 0;

for specIdx = 1:numel(inputSpecs)
    spec = inputSpecs(specIdx);
    fprintf('\n=== Robust simulation | method=%s | source=%s | %d candidates ===\n', ...
        spec.methodName, spec.sourceLabel, height(spec.paretoTable));

    allCandidates = evaluate_robust_candidates( ...
        spec.paretoTable, ...
        spec.methodName, ...
        modulationOrder, ...
        simulationSeeds, ...
        simConfig ...
    );

    outputFile = fullfile(outputResultDir, ...
        sprintf('robust_simulation_%s.xlsx', sanitize_file_stem(spec.methodName)));
    metadataTable = build_metadata_table( ...
        spec.inputFile, ...
        spec.sourceSheet, ...
        outputFile, ...
        spec.methodName, ...
        modulationOrder, ...
        simulationSeeds, ...
        height(allCandidates) ...
    );

    write_simulation_workbook( ...
        allCandidates, ...
        metadataTable, ...
        outputFile, ...
        ALL_CANDIDATES_SHEET_NAME, ...
        METADATA_SHEET_NAME ...
    );

    totalCandidates = totalCandidates + height(allCandidates);
    fprintf('Saved robust simulation results to: %s\n', outputFile);
end

fprintf('\nFinished robust simulation.\n');
fprintf('Methods covered: %d\n', numel(inputSpecs));
fprintf('Total evaluated candidates: %d\n', totalCandidates);

%% -------------------- 本地函数 ----------------
function values = normalize_method_list(value)
    if isempty(value)
        values = {};
        return;
    end

    if ischar(value)
        rawValues = strsplit(value, ',');
    elseif isstring(value)
        rawValues = cellstr(value(:));
    elseif iscell(value)
        rawValues = value(:);
    else
        error('SELECTED_METHODS must be a cell, string, char, or empty value. Got: %s', class(value));
    end

    values = {};
    for idx = 1:numel(rawValues)
        item = strtrim(char(string(rawValues{idx})));
        if ~isempty(item)
            values{end + 1} = item; %#ok<AGROW>
        end
    end
    values = unique(values, 'stable');
end

function inputFiles = discover_input_files(inputMooDir, selectedMethods)
    if ~exist(inputMooDir, 'dir')
        error('MOO input directory not found: %s', inputMooDir);
    end

    inputFiles = {};
    if ~isempty(selectedMethods)
        for idx = 1:numel(selectedMethods)
            candidate = fullfile(inputMooDir, ...
                sprintf('%s.xlsx', sanitize_file_stem(selectedMethods{idx})));
            if exist(candidate, 'file')
                inputFiles{end + 1} = candidate; %#ok<AGROW>
            else
                fprintf('Warning: method file not found: %s\n', candidate);
            end
        end
        return;
    end

    fileInfos = dir(fullfile(inputMooDir, '*.xlsx'));
    for idx = 1:numel(fileInfos)
        fileName = fileInfos(idx).name;
        if startsWith(fileName, '~$')
            continue;
        end
        inputFiles{end + 1} = fullfile(fileInfos(idx).folder, fileName); %#ok<AGROW>
    end
end

function inputSpecs = load_input_specs(inputFiles, inputSheetName)
    inputSpecs = struct( ...
        'methodName', {}, ...
        'paretoTable', {}, ...
        'inputFile', {}, ...
        'sourceSheet', {}, ...
        'sourceLabel', {} ...
    );
    requiredInputColumns = {'x1', 'x2', 'x3', 'x4'};

    for fileIdx = 1:numel(inputFiles)
        inputFile = inputFiles{fileIdx};
        if ~exist(inputFile, 'file')
            fprintf('Warning: skipping missing input file: %s\n', inputFile);
            continue;
        end

        try
            paretoTable = readtable(inputFile, 'Sheet', inputSheetName);
        catch readErr
            fprintf('Warning: failed to read %s [%s]: %s\n', ...
                inputFile, inputSheetName, readErr.message);
            continue;
        end

        if isempty(paretoTable) || height(paretoTable) == 0
            fprintf('Warning: empty Pareto table in %s [%s]\n', inputFile, inputSheetName);
            continue;
        end

        if ~has_required_columns(paretoTable, requiredInputColumns)
            fprintf('Warning: skipping input without required columns: %s\n', inputFile);
            continue;
        end

        paretoTable = ensure_candidate_identity_columns(paretoTable);
        methodName = get_file_stem(inputFile);
        existingMethods = {inputSpecs.methodName};
        if any(strcmp(existingMethods, methodName))
            error('Duplicate method input detected: %s', methodName);
        end

        inputSpecs(end + 1) = struct( ... %#ok<AGROW>
            'methodName', methodName, ...
            'paretoTable', paretoTable, ...
            'inputFile', inputFile, ...
            'sourceSheet', inputSheetName, ...
            'sourceLabel', sprintf('%s :: %s', inputFile, inputSheetName) ...
        );
    end
end

function tf = has_required_columns(tbl, requiredColumns)
    tf = all(ismember(requiredColumns, tbl.Properties.VariableNames));
end

function tbl = ensure_candidate_identity_columns(tbl)
    if ~ismember('moo_run', tbl.Properties.VariableNames)
        tbl.moo_run = ones(height(tbl), 1);
    end
    if ~ismember('solution_idx', tbl.Properties.VariableNames)
        tbl.solution_idx = (1:height(tbl)).';
    end
end

function resultTable = evaluate_robust_candidates( ...
    paretoTable, ...
    methodName, ...
    modulationOrder, ...
    simulationSeeds, ...
    simConfig ...
)
    numRows = height(paretoTable);
    numSeeds = numel(simulationSeeds);

    throughputValues = zeros(numRows, numSeeds);
    berValues = zeros(numRows, numSeeds);
    paprValues = zeros(numRows, numSeeds);

    for rowIdx = 1:numRows
        for seedIdx = 1:numSeeds
            currentSeed = simulationSeeds(seedIdx);
            metrics = ofdm_link_quality_ex( ...
                paretoTable.x1(rowIdx), ...
                paretoTable.x2(rowIdx), ...
                paretoTable.x3(rowIdx), ...
                paretoTable.x4(rowIdx), ...
                modulationOrder, ...
                currentSeed, ...
                simConfig ...
            );

            throughputValues(rowIdx, seedIdx) = metrics.thr_mbps;
            berValues(rowIdx, seedIdx) = metrics.ber;
            paprValues(rowIdx, seedIdx) = metrics.papr_dB;
        end

        if mod(rowIdx, 10) == 0 || rowIdx == numRows
            fprintf('  %s completed %d / %d candidates (%d seeds each)\n', ...
                methodName, rowIdx, numRows, numSeeds);
        end
    end

    resultTable = paretoTable;
    resultTable.method = repmat({methodName}, numRows, 1);

    robustMeanThroughput = mean(throughputValues, 2);
    robustVarThroughput = var(throughputValues, 1, 2);
    robustMeanBer = mean(berValues, 2);
    robustVarBer = var(berValues, 1, 2);
    robustMeanPapr = mean(paprValues, 2);
    robustVarPapr = var(paprValues, 1, 2);

    resultTable.robust_mean_throughput_mbps = robustMeanThroughput;
    resultTable.robust_var_throughput_mbps = robustVarThroughput;
    resultTable.robust_mean_ber = robustMeanBer;
    resultTable.robust_var_ber = robustVarBer;
    resultTable.robust_mean_papr_db = robustMeanPapr;
    resultTable.robust_var_papr_db = robustVarPapr;
end

function metadataTable = build_metadata_table( ...
    inputMooFile, ...
    sourceSheet, ...
    outputResultFile, ...
    methodName, ...
    modulationOrder, ...
    simulationSeeds, ...
    numCandidates ...
)
    metadataTable = table( ...
        {inputMooFile}, ...
        {sourceSheet}, ...
        {outputResultFile}, ...
        {methodName}, ...
        modulationOrder, ...
        {mat2str(simulationSeeds)}, ...
        numel(simulationSeeds), ...
        {'existing AWGN/channel/bit randomness controlled by rngseed'}, ...
        5, ...
        numCandidates, ...
        {'simulation only; robust loss/Pareto/rank are computed by analyze_robust_ofdm_pareto.py'}, ...
        'VariableNames', { ...
            'input_moo_file', ...
            'source_sheet', ...
            'output_result_file', ...
            'method', ...
            'modulation_order', ...
            'simulation_seeds', ...
            'num_repetitions', ...
            'noise_source', ...
            'fixed_noise_figure_db', ...
            'num_candidates', ...
            'postprocess_note' ...
        } ...
    );
end

function write_simulation_workbook( ...
    allCandidates, ...
    metadataTable, ...
    outputFile, ...
    allCandidatesSheetName, ...
    metadataSheetName ...
)
    outputDir = fileparts(outputFile);
    if ~isempty(outputDir) && ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    if exist(outputFile, 'file')
        delete(outputFile);
    end

    writetable(allCandidates, outputFile, 'Sheet', allCandidatesSheetName);
    writetable(metadataTable, outputFile, 'Sheet', metadataSheetName);
end

function fileStem = get_file_stem(filePath)
    [~, fileStem, ~] = fileparts(filePath);
end

function fileStem = sanitize_file_stem(name)
    fileStem = char(name);
    invalidChars = '[]:*?/\<>|"';
    for idx = 1:numel(invalidChars)
        fileStem(fileStem == invalidChars(idx)) = '_';
    end
    fileStem = strtrim(fileStem);
    if isempty(fileStem)
        fileStem = 'results';
    end
end
