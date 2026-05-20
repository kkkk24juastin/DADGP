% PLOT_OFDM_CONSTELLATION_COMPARISON
% 为每个配置方法的选定候选点绘制均衡后的 OFDM 星座图。
%
% 不同方法之间只改变所选工作点。为保证可视化比较公平，OFDM 波形生成、
% 信道、均衡、调制阶数和 RNG 种子均保持一致。
% 选择规则:
%   1) 保留 throughput >= threshold 的候选点。
%   2) 在可行集合内最小化 EVM。
%   3) 依次用 BER、PAPR、throughput 和索引打破平局。
% 每种方法保存为独立图，而不是合并为一个 tiled grid。
%
% 直接修改下方用户配置区域。
% `SELECTED_METHODS = {}` 表示扫描 INPUT_MOO_DIR 下全部可用方法文件。
% 默认优先使用稳健仿真工作簿（如果存在）。

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
SELECTED_METHODS = { ...
    'dadgp', ...
    'baseline_equal', ...
    'baseline_pure_dgp', ...
    'baseline_dwa', ...
    'baseline_uw', ...
    'baseline_mgda', ...
    'baseline_indep_dgp', ...
    'baseline_indep_hetgp', ...
    'baseline_lmc_dgp', ...
    'ablation_no_sample_attn', ...
    'bo_qehvi', ...
    'bo_qnehvi', ...
    'bo_qparego' ...
};
INPUT_MOO_DIR = fullfile(projectDir, 'moo');
INPUT_RESULT_DIR = fullfile(projectDir, 'result');
OUTPUT_FIG_DIR = fullfile(projectDir, 'fig', 'ofdm_constellation');

MODULATION_ORDER = 16;
SIMULATION_SEED = 42;
SIM_CONFIG = [];

% 选项:
%   'robust_if_available' : 优先使用 result/robust_simulation_<method>.xlsx
%   'moo_only'            : 使用 moo/<method>.xlsx
CANDIDATE_SOURCE = 'robust_if_available';

MOO_SHEET_NAME = 'results';
ROBUST_SHEET_NAME = 'all_candidates';
EVM_LOOKUP_FILE = fullfile(INPUT_RESULT_DIR, 'ofdm_evm_all_points.xlsx');
MIN_THROUGHPUT_FOR_EVM_SELECTION_MBPS = 5.44;

MAX_SYMBOLS_PER_FIGURE = Inf;
ERROR_AXIS_LIMIT = 4.0;
% 使用更紧的圆形坐标范围，避免尾部离群点淹没主体分布。
ERROR_CIRCLE_QUANTILE = 0.90;
ERROR_MARKER_SIZE = 6.5;
ERROR_MARKER_ALPHA = 0.28;
EXPORT_BASENAME_PREFIX = 'ofdm_error_constellation_13methods';

% 字体大小与 plot_paper_4obj_pareto.py 保持一致。
TITLE_FONT_SIZE = 22;
LABEL_FONT_SIZE = 20;
TICK_FONT_SIZE = 18;
LEGEND_FONT_SIZE = 18;
EVM_LABEL_FONT_SIZE = 12;

%% -------------------- 配置解析 -----------------
selectedMethods = normalize_method_list(SELECTED_METHODS);
inputMooDir = char(string(INPUT_MOO_DIR));
inputResultDir = char(string(INPUT_RESULT_DIR));
outputFigDir = char(string(OUTPUT_FIG_DIR));
modulationOrder = MODULATION_ORDER;
simulationSeed = SIMULATION_SEED;
simConfig = SIM_CONFIG;
throughputFloorMbps = MIN_THROUGHPUT_FOR_EVM_SELECTION_MBPS;
fontSizes = struct( ...
    'title', TITLE_FONT_SIZE, ...
    'label', LABEL_FONT_SIZE, ...
    'tick', TICK_FONT_SIZE, ...
    'legend', LEGEND_FONT_SIZE, ...
    'evmLabel', EVM_LABEL_FONT_SIZE ...
);

if ~exist(outputFigDir, 'dir')
    mkdir(outputFigDir);
end

methodSpecs = load_method_specs( ...
    selectedMethods, ...
    inputMooDir, ...
    inputResultDir, ...
    CANDIDATE_SOURCE, ...
    MOO_SHEET_NAME, ...
    ROBUST_SHEET_NAME ...
);

if isempty(methodSpecs)
    error('No processable method candidate tables were found.');
end

evmLookupTable = load_evm_lookup_table(EVM_LOOKUP_FILE, selectedMethods);

fprintf('Input MOO dir: %s\n', inputMooDir);
fprintf('Input result dir: %s\n', inputResultDir);
fprintf('Output figure dir: %s\n', outputFigDir);
fprintf('Candidate source: %s\n', CANDIDATE_SOURCE);
fprintf('Modulation order: %d\n', modulationOrder);
fprintf('Simulation seed: %d\n', simulationSeed);
fprintf('Throughput floor for EVM selection: %.2f Mbps\n', throughputFloorMbps);
fprintf('Selected methods: %s\n', strjoin({methodSpecs.methodName}, ', '));

%% -------------------- 选择并仿真 --------------
for specIdx = 1:numel(methodSpecs)
    spec = methodSpecs(specIdx);
    [selectedRow, selectionScore, selectionInfo] = select_representative_candidate( ...
        spec.candidateTable, ...
        evmLookupTable, ...
        throughputFloorMbps ...
    );

    fprintf('\n=== Constellation | method=%s | source=%s ===\n', ...
        spec.methodName, spec.sourceLabel);
    fprintf('Selection rule: %s (%d feasible of %d candidates)\n', ...
        selectionInfo.selectionBasis, selectionInfo.numFeasible, selectionInfo.numCandidates);
    fprintf('Selected x = [%.6g, %.6g, %.6g, %.6g]\n', ...
        selectedRow.x1, selectedRow.x2, selectedRow.x3, selectedRow.x4);
    fprintf('Selected metrics: EVM=%.6g, throughput=%.6g Mbps, BER=%.6g, PAPR=%.6g dB\n', ...
        evm_rms_to_db(selectedRow.real_evm_rms), ...
        selectedRow.evm_eval_throughput_mbps, ...
        selectedRow.evm_eval_ber, ...
        selectedRow.evm_eval_papr_db);

    [metrics, aux] = ofdm_link_quality_ex( ...
        selectedRow.x1, ...
        selectedRow.x2, ...
        selectedRow.x3, ...
        selectedRow.x4, ...
        modulationOrder, ...
        simulationSeed, ...
        simConfig ...
    );

    plotSpec = struct( ...
        'methodName', spec.methodName, ...
        'methodLabel', method_label(spec.methodName), ...
        'methodColor', method_color(spec.methodName), ...
        'sourceLabel', spec.sourceLabel, ...
        'selectionScore', selectionScore, ...
        'selectedRow', selectedRow, ...
        'selectionInfo', selectionInfo, ...
        'metrics', metrics, ...
        'aux', aux ...
    );

    fprintf('Real metrics: throughput=%.4g Mbps, BER=%.4g, PAPR=%.4g dB, EVM=%.4g\n', ...
        metrics.thr_mbps, metrics.ber, metrics.papr_dB, evm_rms_to_db(metrics.evm_rms));
    figureBaseName = sprintf('%s_%s', EXPORT_BASENAME_PREFIX, sanitize_file_stem(spec.methodName));
    figurePathPng = fullfile(outputFigDir, sprintf('%s.png', figureBaseName));
    figurePathPdf = fullfile(outputFigDir, sprintf('%s.pdf', figureBaseName));

    plot_constellation_figure( ...
        plotSpec, ...
        modulationOrder, ...
        simulationSeed, ...
        max(1, MAX_SYMBOLS_PER_FIGURE), ...
        ERROR_AXIS_LIMIT, ...
        ERROR_CIRCLE_QUANTILE, ...
        ERROR_MARKER_SIZE, ...
        ERROR_MARKER_ALPHA, ...
        fontSizes, ...
        figurePathPng, ...
        figurePathPdf ...
    );

    fprintf('Saved constellation figure to:\n');
    fprintf('  %s\n', figurePathPng);
    fprintf('  %s\n', figurePathPdf);
end

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

function methodSpecs = load_method_specs(selectedMethods, inputMooDir, inputResultDir, candidateSource, mooSheetName, robustSheetName)
    inputFiles = discover_moo_files(inputMooDir, selectedMethods);
    methodSpecs = struct( ...
        'methodName', {}, ...
        'candidateTable', {}, ...
        'inputFile', {}, ...
        'sourceSheet', {}, ...
        'sourceLabel', {} ...
    );

    for fileIdx = 1:numel(inputFiles)
        mooFile = inputFiles{fileIdx};
        methodName = get_file_stem(mooFile);
        [candidateTable, inputFile, sourceSheet] = read_candidate_table( ...
            methodName, ...
            mooFile, ...
            inputResultDir, ...
            candidateSource, ...
            mooSheetName, ...
            robustSheetName ...
        );

        if isempty(candidateTable) || height(candidateTable) == 0
            fprintf('Warning: skipping empty candidate table for method=%s\n', methodName);
            continue;
        end
        if ~has_required_columns(candidateTable, {'x1', 'x2', 'x3', 'x4', 'moo_run', 'solution_idx'})
            fprintf('Warning: skipping table without x1-x4/moo_run/solution_idx columns: %s\n', inputFile);
            continue;
        end

        candidateTable.method = repmat(string(methodName), height(candidateTable), 1);

        methodSpecs(end + 1) = struct( ... %#ok<AGROW>
            'methodName', methodName, ...
            'candidateTable', candidateTable, ...
            'inputFile', inputFile, ...
            'sourceSheet', sourceSheet, ...
            'sourceLabel', sprintf('%s :: %s', inputFile, sourceSheet) ...
        );
    end
end

function inputFiles = discover_moo_files(inputMooDir, selectedMethods)
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

function evmLookupTable = load_evm_lookup_table(evmLookupFile, selectedMethods)
    if ~exist(evmLookupFile, 'file')
        error('EVM lookup workbook not found: %s', evmLookupFile);
    end

    evmLookupTable = readtable(evmLookupFile, 'Sheet', 'all_points');
    requiredColumns = { ...
        'method', ...
        'moo_run', ...
        'solution_idx', ...
        'real_evm_rms', ...
        'evm_eval_throughput_mbps', ...
        'evm_eval_ber', ...
        'evm_eval_papr_db' ...
    };
    if ~has_required_columns(evmLookupTable, requiredColumns)
        error('EVM lookup workbook is missing required columns.');
    end

    evmLookupTable.method = string(evmLookupTable.method);
    evmLookupTable = evmLookupTable(ismember(evmLookupTable.method, string(selectedMethods)), :);
    evmLookupTable = evmLookupTable(:, requiredColumns);
end

function [candidateTable, inputFile, sourceSheet] = read_candidate_table(methodName, mooFile, inputResultDir, candidateSource, mooSheetName, robustSheetName)
    robustFile = fullfile(inputResultDir, ...
        sprintf('robust_simulation_%s.xlsx', sanitize_file_stem(methodName)));
    useRobust = strcmp(candidateSource, 'robust_if_available') && exist(robustFile, 'file');

    if useRobust
        inputFile = robustFile;
        sourceSheet = robustSheetName;
    else
        inputFile = mooFile;
        sourceSheet = mooSheetName;
    end

    try
        candidateTable = readtable(inputFile, 'Sheet', sourceSheet);
    catch readErr
        error('Failed to read %s [%s]: %s', inputFile, sourceSheet, readErr.message);
    end
end

function [selectedRow, selectionScore, selectionInfo] = select_representative_candidate(candidateTable, evmLookupTable, throughputFloorMbps)
    mergedTable = innerjoin(candidateTable, evmLookupTable, 'Keys', {'method', 'moo_run', 'solution_idx'});
    if isempty(mergedTable)
        error('No matching EVM rows were found for the candidate table.');
    end

    feasibleMask = mergedTable.evm_eval_throughput_mbps >= throughputFloorMbps;
    selectionInfo = struct( ...
        'throughputFloorMbps', throughputFloorMbps, ...
        'numCandidates', height(mergedTable), ...
        'numFeasible', sum(feasibleMask), ...
        'selectionBasis', '' ...
    );

    if any(feasibleMask)
        eligibleTable = mergedTable(feasibleMask, :);
        selectionInfo.selectionBasis = sprintf('throughput >= %.2f Mbps, then minimum EVM', throughputFloorMbps);
    else
        eligibleTable = mergedTable;
        selectionInfo.selectionBasis = sprintf('no candidate met throughput >= %.2f Mbps, fallback to minimum EVM', throughputFloorMbps);
    end

    sortMatrix = [ ...
        eligibleTable.real_evm_rms, ...
        eligibleTable.evm_eval_ber, ...
        eligibleTable.evm_eval_papr_db, ...
        -eligibleTable.evm_eval_throughput_mbps, ...
        eligibleTable.moo_run, ...
        eligibleTable.solution_idx ...
    ];
    [~, order] = sortrows(sortMatrix);
    selectedRow = eligibleTable(order(1), :);
    selectionScore = selectedRow.real_evm_rms;
end

function plot_constellation_figure(plotSpec, modulationOrder, simulationSeed, maxSymbolsPerFigure, axisLimit, circleQuantile, markerSize, markerAlpha, fontSizes, outputPng, outputPdf)
    set_paper_style(fontSizes);

    fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 3.3, 3.05]);
    ax = axes('Parent', fig);
    hold(ax, 'on');

    txSymbols = plotSpec.aux.txDataSym(:);
    rxSymbols = plotSpec.aux.rxDataSymEQ(:);
    if numel(txSymbols) ~= numel(rxSymbols)
        error('txDataSym and rxDataSymEQ must have the same length for error constellation plotting.');
    end

    scaleRef = sqrt(mean(abs(txSymbols) .^ 2));
    if ~isfinite(scaleRef) || scaleRef <= 0
        scaleRef = 1;
    end
    errorSymbols = (rxSymbols - txSymbols) ./ scaleRef;
    errorSymbols = downsample_symbols(errorSymbols, maxSymbolsPerFigure);

    scatter(ax, 0, 0, 24, [0.08, 0.08, 0.08], 'x', 'LineWidth', 0.9);

    rxHandle = scatter(ax, real(errorSymbols), imag(errorSymbols), markerSize, ...
        plotSpec.methodColor, ...
        'filled', ...
        'MarkerEdgeColor', 'none', ...
        'Marker', 'o');
    try
        rxHandle.MarkerFaceAlpha = markerAlpha;
        rxHandle.MarkerEdgeAlpha = markerAlpha;
    catch
    end

    if ~isempty(errorSymbols)
        rQuantile = prctile(abs(errorSymbols), 100 * circleQuantile);
        theta = linspace(0, 2 * pi, 256);
        plot(ax, rQuantile * cos(theta), rQuantile * sin(theta), ...
            'Color', [0.22, 0.22, 0.22], ...
            'LineStyle', '--', ...
            'LineWidth', 0.8);
    end

    axis(ax, 'equal');
    xlim(ax, [-axisLimit, axisLimit]);
    ylim(ax, [-axisLimit, axisLimit]);
    grid(ax, 'on');
    box(ax, 'off');
    style_axes(ax, fontSizes);

    if strcmp(plotSpec.methodName, 'dadgp')
        ax.LineWidth = 0.85;
    end

    add_evm_label(ax, plotSpec.selectedRow.real_evm_rms, fontSizes);

    xlabel(ax, 'Error In-phase', 'FontSize', fontSizes.label);
    ylabel(ax, 'Error Quadrature', 'FontSize', fontSizes.label);

    save_figure(fig, outputPng, outputPdf);
    close(fig);
end

function add_evm_label(ax, evmRmsValue, fontSizes)
    evmDisplayValue = evm_rms_to_db(evmRmsValue);
    if ~isfinite(evmDisplayValue)
        labelText = 'EVM = N/A';
    else
        labelText = sprintf('EVM = %.3f', evmDisplayValue);
    end

    text(ax, 0.035, 0.955, labelText, ...
        'Units', 'normalized', ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'top', ...
        'FontName', 'Arial', ...
        'FontSize', fontSizes.evmLabel, ...
        'FontWeight', 'bold', ...
        'Color', [0.10, 0.10, 0.10], ...
        'BackgroundColor', [1.00, 1.00, 1.00], ...
        'EdgeColor', [0.72, 0.72, 0.72], ...
        'LineWidth', 0.45, ...
        'Margin', 2.4);
end

function evmDbValue = evm_rms_to_db(evmRmsValue)
    evmDbValue = 20 * log10(max(evmRmsValue, realmin));
end

function symbols = downsample_symbols(symbols, maxSymbols)
    symbols = symbols(:);
    numSymbols = numel(symbols);
    if numSymbols <= maxSymbols
        return;
    end
    indices = unique(round(linspace(1, numSymbols, maxSymbols)));
    symbols = symbols(indices);
end

function set_paper_style(fontSizes)
    set(groot, 'defaultFigureColor', 'w');
    set(groot, 'defaultAxesFontName', 'Arial');
    set(groot, 'defaultTextFontName', 'Arial');
    set(groot, 'defaultAxesFontSize', fontSizes.tick);
    set(groot, 'defaultTextFontSize', fontSizes.tick);
    set(groot, 'defaultAxesLineWidth', 0.55);
    try
        set(groot, 'defaultLegendFontSize', fontSizes.legend);
        set(groot, 'defaultAxesTitleFontSizeMultiplier', fontSizes.title / fontSizes.tick);
        set(groot, 'defaultAxesLabelFontSizeMultiplier', fontSizes.label / fontSizes.tick);
    catch
    end
end

function style_axes(ax, fontSizes)
    ax.FontName = 'Arial';
    ax.FontSize = fontSizes.tick;
    ax.LineWidth = 0.55;
    ax.XColor = [0.17, 0.17, 0.17];
    ax.YColor = [0.17, 0.17, 0.17];
    ax.GridColor = [0.84, 0.84, 0.84];
    ax.GridAlpha = 0.65;
    ax.TickDir = 'out';
    ax.TickLength = [0.015, 0.015];
end

function save_figure(fig, outputPng, outputPdf)
    outputDir = fileparts(outputPng);
    if ~isempty(outputDir) && ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    try
        exportgraphics(fig, outputPng, 'Resolution', 300, 'ContentType', 'image');
        exportgraphics(fig, outputPdf, 'Resolution', 300, 'ContentType', 'image');
    catch
        print(fig, outputPng, '-dpng', '-image', '-r300');
        print(fig, outputPdf, '-dpdf', '-image', '-r300');
    end
end

function label = method_label(methodName)
    switch methodName
        case 'dadgp'
            label = 'DADGP';
        case 'baseline_equal'
            label = 'Equal';
        case 'baseline_pure_dgp'
            label = 'Pure DGP';
        case 'baseline_dwa'
            label = 'DWA';
        case 'baseline_uw'
            label = 'UW';
        case 'baseline_mgda'
            label = 'MGDA';
        case 'baseline_indep_dgp'
            label = 'Indep-DGP';
        case 'baseline_indep_hetgp'
            label = 'Indep-HetGP';
        case 'baseline_lmc_dgp'
            label = 'LMC-DGP';
        case 'ablation_no_sample_attn'
            label = 'No Sample Attn';
        case 'bo_qehvi'
            label = 'BO-qEHVI';
        case 'bo_qnehvi'
            label = 'BO-qNEHVI';
        case 'bo_qparego'
            label = 'BO-qParEGO';
        otherwise
            label = methodName;
    end
end

function color = method_color(methodName)
    switch methodName
        case 'dadgp'
            color = hex2rgb('#B64342');
        case 'baseline_equal'
            color = hex2rgb('#484878');
        case 'baseline_pure_dgp'
            color = hex2rgb('#A8A8A8');
        case 'baseline_dwa'
            color = hex2rgb('#7884B4');
        case 'baseline_uw'
            color = hex2rgb('#9A7FA8');
        case 'baseline_mgda'
            color = hex2rgb('#42949E');
        case 'baseline_indep_dgp'
            color = hex2rgb('#7DA7A1');
        case 'baseline_indep_hetgp'
            color = hex2rgb('#8C7A6B');
        case 'baseline_lmc_dgp'
            color = hex2rgb('#B8A15B');
        case 'ablation_no_sample_attn'
            color = hex2rgb('#606060');
        case 'bo_qehvi'
            color = hex2rgb('#7C6CCF');
        case 'bo_qnehvi'
            color = hex2rgb('#5B8FD6');
        case 'bo_qparego'
            color = hex2rgb('#D08A55');
        otherwise
            color = hex2rgb('#777777');
    end
end

function color = hex2rgb(hexValue)
    hexValue = char(hexValue);
    if startsWith(hexValue, '#')
        hexValue = hexValue(2:end);
    end
    color = [ ...
        hex2dec(hexValue(1:2)), ...
        hex2dec(hexValue(3:4)), ...
        hex2dec(hexValue(5:6)) ...
    ] ./ 255;
end

function tf = has_required_columns(tbl, requiredColumns)
    tf = all(ismember(requiredColumns, tbl.Properties.VariableNames));
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
