function matlab_deepc_file_controller(varargin)
%MATLAB_DEEPC_FILE_CONTROLLER Classical DeePC controller for the ns-3 file bridge.
%
% Example:
%   matlab_deepc_file_controller( ...
%       'BridgeDir', 'C:\Users\bim\deepc_bridge_run01\bridge', ...
%       'DataMat', 'C:\Users\bim\matlab_deepc_data.mat', ...
%       'YalmipDir', 'C:\tools\YALMIP')
%
% The controller watches BridgeDir\requests for request_XXXXXX.json and writes
% matching BridgeDir\responses\response_XXXXXX.json. It uses the classical
% Hankel DeePC decision variable g:
%
%   Up*g ~= u_ini, Yp*g ~= y_ini, Dp*g ~= d_ini, Df*g ~= d_future
%   u_future = Uf*g, y_future = Yf*g

p = inputParser;
addParameter(p, 'BridgeDir', fullfile('data', 'output', 'deepc_matlab_bridge_run01', 'bridge'));
addParameter(p, 'DataMat', fullfile('data', 'output', 'deepc_open_loop_250veh', 'matlab_deepc_data.mat'));
addParameter(p, 'YalmipDir', '');
addParameter(p, 'Solver', 'quadprog');
addParameter(p, 'PollSeconds', 0.05);
addParameter(p, 'IdleTimeoutSeconds', 0);
addParameter(p, 'MaxSteps', 0);
addParameter(p, 'MaxHankelCols', 300);
addParameter(p, 'TargetPrr', 0.95);
addParameter(p, 'TargetPir', 0.10);
addParameter(p, 'TargetCbr', 0.35);
addParameter(p, 'TargetPowerDbm', 16.0);
addParameter(p, 'TargetBeaconIntervalS', 0.10);
addParameter(p, 'WeightPrr', 40.0);
addParameter(p, 'WeightPir', 3.0);
addParameter(p, 'WeightCbr', 8.0);
addParameter(p, 'WeightPower', 0.002);
addParameter(p, 'WeightBeaconInterval', 0.5);
addParameter(p, 'WeightDeltaPower', 0.05);
addParameter(p, 'WeightDeltaBeaconInterval', 15.0);
addParameter(p, 'LambdaIniU', 2.0e3);
addParameter(p, 'LambdaIniY', 2.0e3);
addParameter(p, 'LambdaIniD', 4.0e2);
addParameter(p, 'LambdaFutureD', 4.0e2);
addParameter(p, 'LambdaG', 1.0e-3);
addParameter(p, 'MinPowerDbm', 10.0);
addParameter(p, 'MaxPowerDbm', 23.0);
addParameter(p, 'MinBeaconIntervalS', 0.05);
addParameter(p, 'MaxBeaconIntervalS', 0.50);
addParameter(p, 'UseSumToOne', true);
parse(p, varargin{:});
cfg = p.Results;

if ~isempty(cfg.YalmipDir)
    addpath(genpath(cfg.YalmipDir));
end
assert(exist('sdpvar', 'file') == 2, 'YALMIP is not on the MATLAB path.');

data = load(cfg.DataMat);
requestDir = fullfile(cfg.BridgeDir, 'requests');
responseDir = fullfile(cfg.BridgeDir, 'responses');
if ~exist(responseDir, 'dir')
    mkdir(responseDir);
end

fprintf('MATLAB DeePC controller watching: %s\n', requestDir);
fprintf('Using Hankel data: %s\n', cfg.DataMat);

processed = containers.Map('KeyType', 'char', 'ValueType', 'logical');
completed = 0;
lastActivity = tic;

while true
    files = dir(fullfile(requestDir, 'request_*.json'));
    [~, order] = sort({files.name});
    files = files(order);
    didWork = false;

    for i = 1:numel(files)
        requestName = files(i).name;
        if isKey(processed, requestName)
            continue;
        end

        requestPath = fullfile(files(i).folder, requestName);
        responsePath = response_path_for(responseDir, requestName);
        if exist(responsePath, 'file')
            processed(requestName) = true;
            continue;
        end

        req = jsondecode(fileread(requestPath));
        tStart = tic;
        try
            sol = solve_request(req, data, cfg);
            success = sol.success;
            txPowerDbm = sol.txPowerDbm;
            beaconIntervalS = sol.beaconIntervalS;
            objective = sol.objective;
            solverStatus = sol.solverStatus;
        catch err
            warning('DeePC solve failed at %s: %s', requestName, err.message);
            success = false;
            previousU = as_column(req.previous_u);
            txPowerDbm = previousU(1);
            beaconIntervalS = previousU(2);
            objective = NaN;
            solverStatus = err.message;
        end
        solveTime = toc(tStart);

        payload = struct();
        payload.step = req.step;
        payload.time_s = req.time_s;
        payload.success = success;
        payload.tx_power_dbm = txPowerDbm;
        payload.beacon_interval_s = beaconIntervalS;
        payload.objective = objective;
        payload.solve_time_s = solveTime;
        payload.solver_status = solverStatus;
        payload.controller = 'matlab_yalmip_classical_deepc';
        write_json_atomic(responsePath, payload);

        processed(requestName) = true;
        completed = completed + 1;
        didWork = true;
        lastActivity = tic;
        fprintf('step=%d success=%d u=[%.3f, %.3f] solve=%.3fs\n', ...
            req.step, success, txPowerDbm, beaconIntervalS, solveTime);

        if cfg.MaxSteps > 0 && completed >= cfg.MaxSteps
            return;
        end
    end

    if cfg.IdleTimeoutSeconds > 0 && toc(lastActivity) > cfg.IdleTimeoutSeconds
        return;
    end
    if ~didWork
        pause(cfg.PollSeconds);
    end
end
end

function sol = solve_request(req, data, cfg)
uIni = as_column(req.u_ini);
yIni = as_column(req.y_ini);
dIni = as_column(req.d_ini);
dFuture = as_column(req.d_future);
previousU = as_column(req.previous_u);

nu = numel(previousU);
ny = 3;
nd = 5;
N = req.future_horizon;
H = prepare_hankel_for_request(data, req, nu, ny, nd);

uIniNorm = normalize_blocks(uIni, data.input_mean, data.input_std, nu);
yIniNorm = normalize_blocks(yIni, data.output_mean, data.output_std, ny);
dIniNorm = normalize_blocks(dIni, data.context_mean, data.context_std, nd);
dFutureNorm = normalize_blocks(dFuture, data.context_mean, data.context_std, nd);
previousUNorm = normalize_blocks(previousU, data.input_mean, data.input_std, nu);

idx = select_hankel_columns(H, uIniNorm, yIniNorm, dIniNorm, dFutureNorm, cfg.MaxHankelCols);
Up = H.Up(:, idx);
Uf = H.Uf(:, idx);
Yp = H.Yp(:, idx);
Yf = H.Yf(:, idx);
Dp = H.Dp(:, idx);
Df = H.Df(:, idx);
nG = numel(idx);

g = sdpvar(nG, 1);
uFuture = Uf * g;
yFuture = Yf * g;

yRefPhys = repmat([cfg.TargetPrr; cfg.TargetPir; cfg.TargetCbr], N, 1);
uRefPhys = repmat([cfg.TargetPowerDbm; cfg.TargetBeaconIntervalS], N, 1);
yRef = normalize_blocks(yRefPhys, data.output_mean, data.output_std, ny);
uRef = normalize_blocks(uRefPhys, data.input_mean, data.input_std, nu);

Q = repmat([cfg.WeightPrr; cfg.WeightPir; cfg.WeightCbr], N, 1);
R = repmat([cfg.WeightPower; cfg.WeightBeaconInterval], N, 1);
DuW = repmat([cfg.WeightDeltaPower; cfg.WeightDeltaBeaconInterval], N, 1);

inputLb = normalize_blocks([cfg.MinPowerDbm; cfg.MinBeaconIntervalS], ...
    data.input_mean, data.input_std, nu);
inputUb = normalize_blocks([cfg.MaxPowerDbm; cfg.MaxBeaconIntervalS], ...
    data.input_mean, data.input_std, nu);

constraints = [repmat(inputLb, N, 1) <= uFuture <= repmat(inputUb, N, 1)];
if cfg.UseSumToOne
    constraints = [constraints, sum(g) == 1];
end

du = [uFuture(1:nu) - previousUNorm; uFuture(nu+1:end) - uFuture(1:end-nu)];

objective = ...
    sum(Q .* (yFuture - yRef).^2) + ...
    sum(R .* (uFuture - uRef).^2) + ...
    cfg.LambdaIniU * sumsqr(Up * g - uIniNorm) + ...
    cfg.LambdaIniY * sumsqr(Yp * g - yIniNorm) + ...
    cfg.LambdaIniD * sumsqr(Dp * g - dIniNorm) + ...
    cfg.LambdaFutureD * sumsqr(Df * g - dFutureNorm) + ...
    sum(DuW .* du.^2) + ...
    cfg.LambdaG * sumsqr(g);

ops = sdpsettings('solver', cfg.Solver, 'verbose', 0);
diagnostics = optimize(constraints, objective, ops);

uApplyNorm = value(uFuture(1:nu));
if any(isnan(uApplyNorm)) || diagnostics.problem ~= 0
    uApplyNorm = previousUNorm;
end
uApply = denormalize_blocks(uApplyNorm, data.input_mean, data.input_std, nu);
uApply(1) = min(max(uApply(1), cfg.MinPowerDbm), cfg.MaxPowerDbm);
uApply(2) = min(max(uApply(2), cfg.MinBeaconIntervalS), cfg.MaxBeaconIntervalS);

sol = struct();
sol.success = diagnostics.problem == 0;
sol.txPowerDbm = uApply(1);
sol.beaconIntervalS = uApply(2);
sol.objective = value(objective);
sol.solverStatus = diagnostics.info;
end

function H = prepare_hankel_for_request(data, req, nu, ny, nd)
dataPast = double(data.past_horizon_samples);
dataFuture = double(data.future_horizon_samples);
reqPast = double(req.past_horizon);
reqFuture = double(req.future_horizon);
assert(reqPast <= dataPast, 'Request past_horizon exceeds exported Hankel past horizon.');
assert(reqFuture <= dataFuture, 'Request future_horizon exceeds exported Hankel future horizon.');

pastU = (dataPast - reqPast) * nu + (1:(reqPast * nu));
pastY = (dataPast - reqPast) * ny + (1:(reqPast * ny));
pastD = (dataPast - reqPast) * nd + (1:(reqPast * nd));
futureU = 1:(reqFuture * nu);
futureY = 1:(reqFuture * ny);
futureD = 1:(reqFuture * nd);

H = struct();
H.Up = data.Up(pastU, :);
H.Uf = data.Uf(futureU, :);
H.Yp = data.Yp(pastY, :);
H.Yf = data.Yf(futureY, :);
H.Dp = data.Dp(pastD, :);
H.Df = data.Df(futureD, :);
end

function idx = select_hankel_columns(H, uIni, yIni, dIni, dFuture, maxCols)
nCols = size(H.Up, 2);
if maxCols <= 0 || maxCols >= nCols
    idx = 1:nCols;
    return;
end
score = sum((H.Up - uIni).^2, 1) + ...
    sum((H.Yp - yIni).^2, 1) + ...
    0.25 * sum((H.Dp - dIni).^2, 1) + ...
    0.25 * sum((H.Df - dFuture).^2, 1);
[~, order] = sort(score, 'ascend');
idx = order(1:maxCols);
end

function x = as_column(value)
x = double(value(:));
end

function normValues = normalize_blocks(values, meanValues, stdValues, blockSize)
values = as_column(values);
meanValues = as_column(meanValues);
stdValues = as_column(stdValues);
nBlocks = numel(values) / blockSize;
assert(abs(nBlocks - round(nBlocks)) < 1e-12, 'Invalid block vector length.');
normValues = zeros(size(values));
for k = 1:round(nBlocks)
    span = (k - 1) * blockSize + (1:blockSize);
    normValues(span) = (values(span) - meanValues) ./ stdValues;
end
end

function values = denormalize_blocks(normValues, meanValues, stdValues, blockSize)
normValues = as_column(normValues);
meanValues = as_column(meanValues);
stdValues = as_column(stdValues);
nBlocks = numel(normValues) / blockSize;
assert(abs(nBlocks - round(nBlocks)) < 1e-12, 'Invalid block vector length.');
values = zeros(size(normValues));
for k = 1:round(nBlocks)
    span = (k - 1) * blockSize + (1:blockSize);
    values(span) = normValues(span) .* stdValues + meanValues;
end
end

function responsePath = response_path_for(responseDir, requestName)
suffix = erase(requestName, 'request_');
responsePath = fullfile(responseDir, ['response_', suffix]);
end

function write_json_atomic(path, payload)
tmpPath = [path, '.tmp'];
text = jsonencode(payload, 'PrettyPrint', true);
fid = fopen(tmpPath, 'w');
assert(fid > 0, 'Cannot open response temp file: %s', tmpPath);
fprintf(fid, '%s\n', text);
fclose(fid);
movefile(tmpPath, path, 'f');
end
