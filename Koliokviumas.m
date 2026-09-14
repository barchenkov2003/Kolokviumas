clear;
clc;
close all;

rng(42);

%% 1. Parametrai

fileName = 'household_power_consumption.txt';

windowMinutes = 15;

trainRatio = 0.60;
valRatio = 0.20;

baselineWindow = 96;

anomalyRate = 0.01;

maxWarningsPerDay = 5;

%% 2. Duomenu nuskaitymas

fprintf('Kraunami duomenys...\n');

opts = detectImportOptions(fileName, ...
    'Delimiter', ';');

opts = setvaropts(opts, ...
    {'Date','Time'}, 'Type', 'string');

T = readtable(fileName, opts);

fprintf('Pradinis irasu skaicius: %d\n', height(T));

%% 3. Datos ir laiko sujungimas

datetimeData = datetime( ...
    T.Date + " " + T.Time, ...
    'InputFormat', 'dd/MM/yyyy HH:mm:ss');

T.DateTime = datetimeData;

T.Date = [];
T.Time = [];

%% 4. Tekstiniu reiksmiu pavertimas i skaicius

for i = 1:width(T)

    if iscell(T{:,i}) || isstring(T{:,i})

        try
            T{:,i} = str2double(T{:,i});
        catch
        end

    end

end

%% 5. Trukstamu reiksmiu tvarkymas

fprintf('Tvarkomos trukstamos reiksmes...\n');

numericNames = T.Properties.VariableNames;

numericNames(strcmp( ...
    numericNames, 'DateTime')) = [];

for i = 1:length(numericNames)

    name = numericNames{i};

    if isnumeric(T.(name))

        T.(name) = fillmissing( ...
            T.(name), ...
            'linear');

        T.(name) = fillmissing( ...
            T.(name), ...
            'previous');

        T.(name) = fillmissing( ...
            T.(name), ...
            'next');

    end

end

%% 6. Duomenu rikiavimas

T = sortrows(T, 'DateTime');

%% 7. Agregavimas i 15 minuciu langus

fprintf('Atliekamas 15 minuciu agregavimas...\n');

TT = table2timetable(T, ...
    'RowTimes', 'DateTime');

TT15 = retime(TT, 'regular', 'mean', ...
    'TimeStep', minutes(windowMinutes));

TT15 = rmmissing(TT15);

fprintf('Irasu po agregavimo: %d\n', height(TT15));

%% 8. Pozyymiu sudarymas

X = [];

X(:,1) = TT15.Global_active_power;

X(:,2) = TT15.Global_reactive_power;

X(:,3) = TT15.Voltage;

X(:,4) = TT15.Global_intensity;

X(:,5) = TT15.Sub_metering_1;

X(:,6) = TT15.Sub_metering_2;

X(:,7) = TT15.Sub_metering_3;

% Aktyvios galios pokytis
activeChange = [0; diff(X(:,1))];

% Aktyvios galios standartinis nuokrypis
activeStd = movstd( ...
    X(:,1), ...
    [3 0]);

X(:,8) = activeStd;

X(:,9) = activeChange;

validRows = all(isfinite(X), 2);

X = X(validRows, :);

timeData = TT15.Time(validRows);

%% 9. Chronologinis skaidymas

N = size(X,1);

trainEnd = floor(N * trainRatio);

valEnd = floor(N * ...
    (trainRatio + valRatio));

X_train = X(1:trainEnd,:);

X_val = X(trainEnd+1:valEnd,:);

X_test = X(valEnd+1:end,:);

time_train = timeData(1:trainEnd);

time_val = timeData(trainEnd+1:valEnd);

time_test = timeData(valEnd+1:end);

fprintf('\nDuomenu skaidymas:\n');

fprintf('Mokymas: %d\n', size(X_train,1));

fprintf('Validacija: %d\n', size(X_val,1));

fprintf('Testas: %d\n', size(X_test,1));

%% 10. Dirbtines anomalijos VALIDACIJAI

[X_val_anomaly, labels_val] = ...
    injectAnomalies(X_val, anomalyRate);

%% 11. Dirbtines anomalijos TESTUI

[X_test_anomaly, labels_test] = ...
    injectAnomalies(X_test, anomalyRate);

%% 12. BASELINE
% Moving Average + Z-score

fprintf('\nBASELINE: Moving Average + Z-score\n');

baselineValScore = ...
    baselineScore(X_val_anomaly, baselineWindow);

baselineTestScore = ...
    baselineScore(X_test_anomaly, baselineWindow);

baselineThreshold = ...
    selectThreshold( ...
        baselineValScore, ...
        labels_val, ...
        maxWarningsPerDay);

baselineMetrics = ...
    calculateMetrics( ...
        baselineTestScore, ...
        labels_test, ...
        baselineThreshold);

fprintf('Threshold: %.4f\n', ...
    baselineThreshold);

fprintf('Recall: %.4f\n', ...
    baselineMetrics.recall);

fprintf('Warnings/day: %.2f\n', ...
    baselineMetrics.warningsPerDay);

%% 13. Duomenu standartizavimas

mu = mean(X_train, 1);

sigma = std(X_train, 0, 1);

sigma(sigma == 0) = 1;

X_train_s = ...
    (X_train - mu) ./ sigma;

X_val_s = ...
    (X_val_anomaly - mu) ./ sigma;

X_test_s = ...
    (X_test_anomaly - mu) ./ sigma;

%% 14. ONE-CLASS SVM

fprintf('\nONE-CLASS SVM\n');

% Del skaiciavimo greicio apribojame mokymo imti

maxSamples = min(30000, size(X_train_s,1));

sampleIndex = randperm( ...
    size(X_train_s,1), ...
    maxSamples);

X_svm_train = ...
    X_train_s(sampleIndex,:);

% Vienos klases SVM

svmModel = fitcsvm( ...
    X_svm_train, ...
    ones(size(X_svm_train,1),1), ...
    'KernelFunction', 'rbf', ...
    'KernelScale', 'auto', ...
    'OutlierFraction', 0.01, ...
    'Standardize', false);

% Score validacijoje

[~, svmValScore] = ...
    predict(svmModel, X_val_s);

% Score teste

[~, svmTestScore] = ...
    predict(svmModel, X_test_s);

% MATLAB score zenklas:
% mazesne reiksme rodo didesne anomalijos tikimybe

svmValScore = -svmValScore;

svmTestScore = -svmTestScore;

%% 15. SVM threshold

svmThreshold = ...
    selectThreshold( ...
        svmValScore, ...
        labels_val, ...
        maxWarningsPerDay);

svmMetrics = ...
    calculateMetrics( ...
        svmTestScore, ...
        labels_test, ...
        svmThreshold);

fprintf('Threshold: %.4f\n', ...
    svmThreshold);

fprintf('Recall: %.4f\n', ...
    svmMetrics.recall);

fprintf('Warnings/day: %.2f\n', ...
    svmMetrics.warningsPerDay);

%% 16. MLP AUTOENCODER

fprintf('\nMLP AUTOENCODER\n');

% Naudojame MLP tinkla rekonstrukcijai

inputSize = size(X_train_s,2);

hiddenSize = 8;

autoencoder = fitnet( ...
    [hiddenSize 3 hiddenSize], ...
    'trainscg');

% Pasleptu sluoksniu aktyvacija

autoencoder.layers{1}.transferFcn = ...
    'tansig';

autoencoder.layers{2}.transferFcn = ...
    'tansig';

autoencoder.layers{3}.transferFcn = ...
    'tansig';

% Isejimas turi atkurti iejima

autoencoder.layers{4}.transferFcn = ...
    'purelin';

% Nenaudojame papildomo duomenu dalinimo

autoencoder.divideFcn = ...
    'dividetrain';

autoencoder.trainParam.epochs = 200;

autoencoder.trainParam.goal = 1e-5;

autoencoder.trainParam.showWindow = true;

% Autoencoder mokosi:
% X -> X

autoencoder = train( ...
    autoencoder, ...
    X_train_s', ...
    X_train_s');

%% 17. Autoencoder anomaliju balas

AE_val_prediction = ...
    autoencoder(X_val_s');

AE_test_prediction = ...
    autoencoder(X_test_s');

% Rekonstrukcijos paklaida

aeValScore = mean( ...
    (X_val_s' - AE_val_prediction).^2, ...
    1)';

aeTestScore = mean( ...
    (X_test_s' - AE_test_prediction).^2, ...
    1)';

%% 18. Autoencoder threshold

aeThreshold = ...
    selectThreshold( ...
        aeValScore, ...
        labels_val, ...
        maxWarningsPerDay);

aeMetrics = ...
    calculateMetrics( ...
        aeTestScore, ...
        labels_test, ...
        aeThreshold);

fprintf('Threshold: %.4f\n', ...
    aeThreshold);

fprintf('Recall: %.4f\n', ...
    aeMetrics.recall);

fprintf('Warnings/day: %.2f\n', ...
    aeMetrics.warningsPerDay);

%% 19. Stabilumas

baselineStability = ...
    calculateStability( ...
        baselineTestScore, ...
        baselineThreshold);

svmStability = ...
    calculateStability( ...
        svmTestScore, ...
        svmThreshold);

aeStability = ...
    calculateStability( ...
        aeTestScore, ...
        aeThreshold);

%% 20. Galutine rezultatu lentele

Method = [
    "Moving Average + Z-score";
    "One-Class SVM";
    "MLP Autoencoder"
    ];

Recall = [
    baselineMetrics.recall;
    svmMetrics.recall;
    aeMetrics.recall
    ];

WarningsPerDay = [
    baselineMetrics.warningsPerDay;
    svmMetrics.warningsPerDay;
    aeMetrics.warningsPerDay
    ];

Stability = [
    baselineStability;
    svmStability;
    aeStability
    ];

Results = table( ...
    Method, ...
    Recall, ...
    WarningsPerDay, ...
    Stability);

fprintf('\n====================================\n');

fprintf('GALUTINIAI REZULTATAI\n');

fprintf('====================================\n');

disp(Results);

writetable( ...
    Results, ...
    'rezultatai.csv');

%% 21. BASELINE grafikas

figure;

plot( ...
    time_test, ...
    baselineTestScore);

hold on;

yline( ...
    baselineThreshold, ...
    '--');

xlabel('Laikas');

ylabel('Anomalijos balas');

title('Moving Average + Z-score');

grid on;

%% 22. SVM grafikas

figure;

plot( ...
    time_test, ...
    svmTestScore);

hold on;

yline( ...
    svmThreshold, ...
    '--');

xlabel('Laikas');

ylabel('Anomalijos balas');

title('One-Class SVM');

grid on;

%% 23. Autoencoder grafikas

figure;

plot( ...
    time_test, ...
    aeTestScore);

hold on;

yline( ...
    aeThreshold, ...
    '--');

xlabel('Laikas');

ylabel('Rekonstrukcijos paklaida');

title('MLP Autoencoder');

grid on;

%% 24. Tikros dirbtines anomalijos

figure;

plot( ...
    time_test, ...
    X_test(:,1));

hold on;

anomalyIndex = ...
    labels_test == 1;

plot( ...
    time_test(anomalyIndex), ...
    X_test(anomalyIndex,1), ...
    'rx');

xlabel('Laikas');

ylabel('Global Active Power');

title('Testo duomenys ir dirbtines anomalijos');

legend( ...
    'Elektros vartojimas', ...
    'Dirbtine anomalija');

grid on;

%% 25. ABLACIJA
% Tik Global Active Power

fprintf('\nABLACIJA\n');

X_train_ab = X_train(:,1);

X_val_ab = X_val_anomaly(:,1);

X_test_ab = X_test_anomaly(:,1);

% Standartizavimas

mu_ab = mean(X_train_ab);

sigma_ab = std(X_train_ab);

if sigma_ab == 0
    sigma_ab = 1;
end

X_train_ab = ...
    (X_train_ab - mu_ab) ./ sigma_ab;

X_val_ab = ...
    (X_val_ab - mu_ab) ./ sigma_ab;

X_test_ab = ...
    (X_test_ab - mu_ab) ./ sigma_ab;

% SVM

svmAb = fitcsvm( ...
    X_train_ab, ...
    ones(size(X_train_ab,1),1), ...
    'KernelFunction', 'rbf', ...
    'KernelScale', 'auto', ...
    'OutlierFraction', 0.01);

[~, scoreAb] = ...
    predict(svmAb, X_val_ab);

scoreAb = -scoreAb;

abThreshold = ...
    selectThreshold( ...
        scoreAb, ...
        labels_val, ...
        maxWarningsPerDay);

[~, scoreAbTest] = ...
    predict(svmAb, X_test_ab);

scoreAbTest = -scoreAbTest;

abMetrics = ...
    calculateMetrics( ...
        scoreAbTest, ...
        labels_test, ...
        abThreshold);

fprintf( ...
    'SVM tik su Global Active Power recall: %.4f\n', ...
    abMetrics.recall);

%% 26. ATSparumo TESTAS

fprintf('\nATSPARUMO TESTAS\n');

% Pridedamas 5 procentu Gaussian triuksmas

noiseLevel = 0.05;

noise = ...
    noiseLevel * randn(size(X_test));

X_test_noisy = ...
    X_test + noise;

X_test_noisy_s = ...
    (X_test_noisy - mu) ./ sigma;

[~, noisyScore] = ...
    predict(svmModel, X_test_noisy_s);

noisyScore = -noisyScore;

noisyMetrics = ...
    calculateMetrics( ...
        noisyScore, ...
        labels_test, ...
        svmThreshold);

fprintf( ...
    'SVM recall su triuksmu: %.4f\n', ...
    noisyMetrics.recall);

fprintf( ...
    'SVM warnings/day su triuksmu: %.2f\n', ...
    noisyMetrics.warningsPerDay);

%% 27. Pabaiga

fprintf('\n====================================\n');

fprintf('ANALIZE BAIGTA\n');

fprintf('Rezultatai issaugoti faile rezultatai.csv\n');

fprintf('====================================\n');


%% FUNKCIJOS

function score = baselineScore(X, W)

    signal = X(:,1);

    movingMean = ...
        movmean(signal, [W 1], ...
        'Endpoints', 'shrink');

    movingStd = ...
        movstd(signal, [W 1], ...
        'Endpoints', 'shrink');

    score = ...
        abs(signal - movingMean) ...
        ./ (movingStd + 1e-8);

end


function threshold = selectThreshold( ...
    scores, labels, maxWarningsPerDay)

    thresholds = ...
        prctile(scores, ...
        linspace(50,99.9,300));

    bestRecall = -1;

    threshold = thresholds(end);

    for i = 1:length(thresholds)

        currentThreshold = ...
            thresholds(i);

        prediction = ...
            scores >= currentThreshold;

        warningsPerDay = ...
            sum(prediction) ...
            / length(prediction) * 96;

        if warningsPerDay <= ...
                maxWarningsPerDay

            TP = sum( ...
                prediction == 1 & ...
                labels == 1);

            FN = sum( ...
                prediction == 0 & ...
                labels == 1);

            if TP + FN > 0

                recall = ...
                    TP / (TP + FN);

            else

                recall = 0;

            end

            if recall > bestRecall

                bestRecall = recall;

                threshold = ...
                    currentThreshold;

            end

        end

    end

end


function metrics = calculateMetrics( ...
    scores, labels, threshold)

    prediction = ...
        scores >= threshold;

    TP = sum( ...
        prediction == 1 & ...
        labels == 1);

    FN = sum( ...
        prediction == 0 & ...
        labels == 1);

    if TP + FN > 0

        recall = ...
            TP / (TP + FN);

    else

        recall = 0;

    end

    days = length(scores) / 96;

    warningsPerDay = ...
        sum(prediction) / days;

    metrics.recall = recall;

    metrics.warningsPerDay = ...
        warningsPerDay;

    metrics.warnings = ...
        sum(prediction);

end


function stability = calculateStability( ...
    scores, threshold)

    prediction = ...
        scores >= threshold;

    dayLength = 96;

    numberOfDays = ...
        floor(length(prediction) / dayLength);

    dailyWarnings = zeros( ...
        numberOfDays, 1);

    for i = 1:numberOfDays

        first = ...
            (i-1)*dayLength + 1;

        last = ...
            i*dayLength;

        dailyWarnings(i) = ...
            sum(prediction(first:last));

    end

    if length(dailyWarnings) > 1

        stability = std(dailyWarnings);

    else

        stability = 0;

    end

end


function [X_new, labels] = ...
    injectAnomalies(X, rate)

    X_new = X;

    N = size(X,1);

    anomalyCount = ...
        max(1, floor(N * rate));

    indices = randperm( ...
        N, anomalyCount);

    labels = zeros(N,1);

    for i = 1:length(indices)

        index = indices(i);

        type = mod(i-1,3);

        if type == 0

            % Staigus vartojimo padidejimas

            X_new(index,1) = ...
                X_new(index,1) * 3;

        elseif type == 1

            % Staigus vartojimo sumazejimas

            X_new(index,1) = ...
                X_new(index,1) * 0.1;

        else

            % Vidutinis vartojimo pokytis

            X_new(index,1) = ...
                X_new(index,1) * 1.8;

        end

        labels(index) = 1;

    end

end