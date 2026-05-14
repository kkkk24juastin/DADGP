% GENERATE_MOO_REAL_SIMULATION_RESULTS
% 基于当前仓库中的 OFDM 仿真，对 `moo/` 下按 method 分开的 Pareto 解集
% 逐点计算真实 throughput / BER / PAPR，并按方法分别导出到 `result/`。
%
% 直接修改下方 User Config 中的配置。
% `SELECTED_METHODS = {}` 表示扫描 `INPUT_MOO_DIR` 下全部方法文件。
% `SELECTED_METHODS = {'dadgp'}` 表示只更新一个方法。

clearvars;
clc;

%% -------------------- Path Setup ---------------------
scriptPath = mfilename('fullpath');
if isempty(scriptPath)
    projectDir = pwd;
else
    projectDir = fileparts(scriptPath);
end

%% -------------------- User Config --------------------
SELECTED_METHODS = {'dadgp'};
INPUT_MOO_DIR = fullfile(projectDir, 'moo');
OUTPUT_RESULT_DIR = fullfile(projectDir, 'result');
MODULATION_ORDER = 16;
SIMULATION_SEED = 42;
SIM_CONFIG = [];
RESULT_SHEET_NAME = 'results';
METADATA_SHEET_NAME = 'metadata';

%% -------------------- Resolve Config -----------------
selectedMethods = normalize_method_list(SELECTED_METHODS);
inputMooDir = char(string(INPUT_MOO_DIR));
outputResultDir = char(string(OUTPUT_RESULT_DIR));
modulationOrder = MODULATION_ORDER;
simulationSeed = SIMULATION_SEED;
simConfig = SIM_CONFIG;

if ~exist(outputResultDir, 'dir')
    mkdir(outputResultDir);
end

inputFiles = discover_input_files(inputMooDir, selectedMethods);
if isempty(inputFiles)
    error('没有找到可用的 MOO 输入文件。');
end

fprintf('Input MOO dir: %s\n', inputMooDir);
fprintf('Output result dir: %s\n', outputResultDir);
fprintf('Modulation order: %d\n', modulationOrder);
fprintf('Simulation seed: %d\n', simulationSeed);
if isempty(selectedMethods)
    fprintf('Selected methods: <all available>\n');
else
    fprintf('Selected methods: %s\n', strjoin(selectedMethods, ', '));
end

%% -------------------- Load Inputs --------------------
inputSpecs = load_input_specs(inputFiles, RESULT_SHEET_NAME);
if isempty(inputSpecs)
    error('没有在输入文件中找到可处理的 Pareto 解。');
end

%% -------------------- Run Simulation -----------------
totalSolutions = 0;

for specIdx = 1:numel(inputSpecs)
    spec = inputSpecs(specIdx);
    fprintf('\n=== Simulating method=%s | source=%s | %d Pareto solutions ===\n', ...
        spec.methodName, spec.sourceLabel, height(spec.paretoTable));

    methodResults = simulate_pareto_table( ...
        spec.paretoTable, ...
        spec.methodName, ...
        modulationOrder, ...
        simulationSeed, ...
        simConfig ...
    );

    outputFile = fullfile(outputResultDir, ...
        sprintf('%s.xlsx', sanitize_file_stem(spec.methodName)));

    metadataTable = build_metadata_table( ...
        spec.inputFile, ...
        spec.sourceSheet, ...
        outputFile, ...
        spec.methodName, ...
        modulationOrder, ...
        simulationSeed, ...
        height(methodResults) ...
    );

    write_method_result_workbook( ...
        methodResults, ...
        metadataTable, ...
        outputFile, ...
        RESULT_SHEET_NAME, ...
        METADATA_SHEET_NAME ...
    );

    totalSolutions = totalSolutions + height(methodResults);
    fprintf('Saved %s results to: %s\n', spec.methodName, outputFile);
end

fprintf('\nFinished real-simulation evaluation.\n');
fprintf('Methods covered: %d\n', numel(inputSpecs));
fprintf('Total evaluated Pareto solutions: %d\n', totalSolutions);

%% -------------------- Local Functions ----------------
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
        error('SELECTED_METHODS 必须为 cell / string / char。当前类型: %s', class(value));
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
        error('未找到 MOO 输入目录: %s', inputMooDir);
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

function inputSpecs = load_input_specs(inputFiles, resultSheetName)
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
            paretoTable = readtable(inputFile, 'Sheet', resultSheetName);
        catch readErr
            fprintf('Warning: failed to read %s [%s]: %s\n', ...
                inputFile, resultSheetName, readErr.message);
            continue;
        end

        if isempty(paretoTable) || height(paretoTable) == 0
            fprintf('Warning: empty Pareto table in %s [%s]\n', inputFile, resultSheetName);
            continue;
        end

        if ~has_required_columns(paretoTable, requiredInputColumns)
            fprintf('Warning: skipping input without required columns: %s\n', inputFile);
            continue;
        end

        methodName = get_file_stem(inputFile);
        existingMethods = {inputSpecs.methodName};
        if any(strcmp(existingMethods, methodName))
            error('检测到重复方法输入: %s。请避免重复提供同一方法的文件。', methodName);
        end

        inputSpecs(end + 1) = struct( ... %#ok<AGROW>
            'methodName', methodName, ...
            'paretoTable', paretoTable, ...
            'inputFile', inputFile, ...
            'sourceSheet', resultSheetName, ...
            'sourceLabel', sprintf('%s :: %s', inputFile, resultSheetName) ...
        );
    end
end

function tf = has_required_columns(tbl, requiredColumns)
    tf = all(ismember(requiredColumns, tbl.Properties.VariableNames));
end

function fileStem = get_file_stem(filePath)
    [~, fileStem, ~] = fileparts(filePath);
end

function resultTable = simulate_pareto_table(paretoTable, methodName, modulationOrder, simulationSeed, simConfig)
    numRows = height(paretoTable);

    throughput = zeros(numRows, 1);
    ber = zeros(numRows, 1);
    papr = zeros(numRows, 1);
    snrDb = zeros(numRows, 1);
    evmRms = zeros(numRows, 1);
    qualityScore = zeros(numRows, 1);
    txStress = zeros(numRows, 1);
    clipRatioDb = zeros(numRows, 1);
    rawRateMbps = zeros(numRows, 1);

    x1 = paretoTable.x1;
    x2 = paretoTable.x2;
    x3 = paretoTable.x3;
    x4 = paretoTable.x4;

    for rowIdx = 1:numRows
        [metrics, aux] = ofdm_link_quality_ex( ...
            x1(rowIdx), ...
            x2(rowIdx), ...
            x3(rowIdx), ...
            x4(rowIdx), ...
            modulationOrder, ...
            simulationSeed, ...
            simConfig ...
        );

        throughput(rowIdx) = metrics.thr_mbps;
        ber(rowIdx) = metrics.ber;
        papr(rowIdx) = metrics.papr_dB;
        snrDb(rowIdx) = metrics.snr_dB;
        evmRms(rowIdx) = metrics.evm_rms;
        qualityScore(rowIdx) = aux.quality_score;
        txStress(rowIdx) = aux.tx_stress;
        clipRatioDb(rowIdx) = aux.clip_ratio_dB;
        rawRateMbps(rowIdx) = aux.raw_rate_mbps;

        if mod(rowIdx, 25) == 0 || rowIdx == numRows
            fprintf('  %s completed %d / %d\n', methodName, rowIdx, numRows);
        end
    end

    [energyPerBit, energyEfficiency] = derive_energy_metrics(x1, throughput);
    realObjectiveTask1 = -throughput;
    realObjectiveTask2 = ber;
    realObjectiveTask3 = papr;
    realTotalObjective = realObjectiveTask1 + realObjectiveTask2 + realObjectiveTask3;

    resultTable = paretoTable;
    resultTable.method = repmat({methodName}, numRows, 1);
    resultTable.throughput_mbps = throughput;
    resultTable.ber = ber;
    resultTable.papr_db = papr;
    resultTable.energy_per_bit = energyPerBit;
    resultTable.energy_efficiency = energyEfficiency;
    resultTable.real_snr_db = snrDb;
    resultTable.real_evm_rms = evmRms;
    resultTable.real_quality_score = qualityScore;
    resultTable.real_tx_stress = txStress;
    resultTable.real_clip_ratio_db = clipRatioDb;
    resultTable.real_raw_rate_mbps = rawRateMbps;
    resultTable.real_objective_task1 = realObjectiveTask1;
    resultTable.real_objective_task2 = realObjectiveTask2;
    resultTable.real_objective_task3 = realObjectiveTask3;
    resultTable.real_total_objective = realTotalObjective;
    resultTable.real_quality_loss_task1 = realObjectiveTask1;
    resultTable.real_quality_loss_task2 = realObjectiveTask2;
    resultTable.real_quality_loss_task3 = realObjectiveTask3;
    resultTable.real_total_quality_loss = realTotalObjective;
    resultTable.modulation_order = repmat(modulationOrder, numRows, 1);
    resultTable.simulation_seed = repmat(simulationSeed, numRows, 1);

    resultTable = sort_result_table(resultTable);
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

function sortedTable = sort_result_table(tbl)
    sortColumns = {};
    candidateColumns = {'method', 'moo_run', 'solution_idx'};
    for idx = 1:numel(candidateColumns)
        if ismember(candidateColumns{idx}, tbl.Properties.VariableNames)
            sortColumns{end + 1} = candidateColumns{idx}; %#ok<AGROW>
        end
    end

    if isempty(sortColumns)
        sortedTable = tbl;
    else
        sortedTable = sortrows(tbl, sortColumns);
    end
end

function metadataTable = build_metadata_table(inputMooFile, sourceSheet, outputResultFile, methodName, modulationOrder, simulationSeed, numSolutions)
    metadataTable = table( ...
        {inputMooFile}, ...
        {sourceSheet}, ...
        {outputResultFile}, ...
        {methodName}, ...
        modulationOrder, ...
        simulationSeed, ...
        numSolutions, ...
        'VariableNames', { ...
            'input_moo_file', ...
            'source_sheet', ...
            'output_result_file', ...
            'method', ...
            'modulation_order', ...
            'simulation_seed', ...
            'num_solutions' ...
        } ...
    );
end

function write_method_result_workbook(resultTable, metadataTable, outputFile, resultSheetName, metadataSheetName)
    outputDir = fileparts(outputFile);
    if ~isempty(outputDir) && ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    if exist(outputFile, 'file')
        delete(outputFile);
    end

    writetable(resultTable, outputFile, 'Sheet', resultSheetName);
    writetable(metadataTable, outputFile, 'Sheet', metadataSheetName);
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
