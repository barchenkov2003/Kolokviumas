%% Namų elektros vartojimo anomalijų aptikimas
% Metodai:
% 1. Moving Average + Z-score
% 2. One-Class SVM
% 3. MLP Autoencoder

clear;
clc;
close all;

%% 1. Duomenu nuskaitymas

filePath = 'C:\Users\barch\OneDrive\Рабочий стол\github\koliokvium\household_power_consumption.txt';

fprintf('Kraunami duomenys...\n');

opts = detectImportOptions(filePath, ...
    'Delimiter', ';', ...
    'DecimalSeparator', '.');

T = readtable(filePath, opts);

fprintf('Pradinis irasu skaicius: %d\n', height(T));

%% 2. Datos ir laiko sutvarkymas

dateText = string(T.Date);
timeText = string(T.Time);

T.DateTime = datetime( ...
    dateText + " " + timeText, ...
    'InputFormat', 'dd/MM/yyyy HH:mm:ss');

%% 3. Trukstamu reiksmiu tvarkymas

fprintf('\nTvarkomos trukstamos reiksmes...\n');

featureNames = { ...
    'Global_active_power', ...
    'Global_reactive_power', ...
    'Voltage', ...
    'Global_intensity', ...
    'Sub_metering_1', ...
    'Sub_metering_2', ...
    'Sub_metering_3'};

for i = 1:length(featureNames)

    columnData = T.(featureNames{i});

    columnData = fillmissing(columnData, 'linear');
    columnData = fillmissing(columnData, 'previous');
    columnData = fillmissing(columnData, 'next');

    T.(featureNames{i}) = columnData;

end

%% 4. Duomenu agregavimas i 15 minuciu intervalus

fprintf('\nAtliekamas 15 minuciu agregavimas...\n');

TT = table2timetable(T);

TT15 = retime(TT, 'regular', 'mean', ...
    'TimeStep', minutes(15));

fprintf('Irasu po agregavimo: %d\n', height(TT15));

%% 5. Trukstamu reiksmiu tvarkymas po agregavimo
% Kai kurie 15 minuciu intervalai gali netureti duomenu.

for i = 1:length(featureNames)

    columnData = TT15.(featureNames{i});

    columnData = fillmissing(columnData, 'linear');
    columnData = fillmissing(columnData, 'previous');
    columnData = fillmissing(columnData, 'next');

    TT15.(featureNames{i}) = columnData;

end

%% 6. Pozymiu sudarymas

activePower = TT15.Global_active_power;
reactivePower = TT15.Global_reactive_power;
voltage = TT15.Voltage;
current = TT15.Global_intensity;

sub1 = TT15.Sub_metering_1;
sub2 = TT15.Sub_metering_2;
sub3 = TT15.Sub_metering_3;

% Aktyvios galios slenkantis standartinis nuokrypis
activeStd = movstd(activePower, [3 0]);

% Aktyvios galios pokytis
activeChange = [0; diff(activePower)];

% Galutine pozymiu matrica
X = [ ...
    activePower, ...
    reactivePower, ...
    voltage, ...
    current, ...
    sub1, ...
    sub2, ...
    sub3, ...
    activeStd, ...
    activeChange];

% Galutinis apsauginis NaN tikrinimas
validRows = all(isfinite(X), 2);

X = X(validRows,:);

fprintf('Galutinis duomenu kiekis: %d\n', size(X,1));

%% 7. Chronologinis duomenu skaidymas

N = size(X,1);

N_train = floor(0.60 * N);
N_validation = floor(0.20 * N);

trainIdx = 1:N_train;

validationIdx = ...
    N_train + 1 : N_train + N_validation;

testIdx = ...
    N_train + N_validation + 1 : N;

X_train = X(trainIdx,:);
X_validation = X(validationIdx,:);
X_test = X(testIdx,:);

fprintf('\nDuomenu skaidymas:\n');
fprintf('Mokymas: %d\n', size(X_train,1));
fprintf('Validacija: %d\n', size(X_validation,1));
fprintf('Testas: %d\n', size(X_test,1));

%% 8. Dirbtiniu anomaliju generavimas

fprintf('\nGeneruojamos dirbtines anomalijos...\n');

X_validation_anomaly = X_validation;
X_test_anomaly = X_test;

labels_validation = zeros(size(X_validation,1),1);
labels_test = zeros(size(X_test,1),1);

% 4 x 15 min = 1 valanda
anomalyLength = 4;

% Apie 1 procentas anomaliju
numValidationAnomalies = ...
    floor(0.01 * size(X_validation,1) / anomalyLength);

numTestAnomalies = ...
    floor(0.01 * size(X_test,1) / anomalyLength);

rng(42);

%% 8.1 Validacijos anomalijos

usedStarts = [];

for k = 1:numValidationAnomalies

    while true

        startIdx = randi( ...
            [1, size(X_validation_anomaly,1) - anomalyLength + 1]);

        if isempty(usedStarts) || ...
                all(abs(startIdx - usedStarts) >= anomalyLength)

            break;

        end

    end

    usedStarts(end+1) = startIdx;

    idx = startIdx:startIdx + anomalyLength - 1;

    % Padidiname aktyviaja galia
    X_validation_anomaly(idx,1) = ...
        X_validation_anomaly(idx,1) * 2.5;

    labels_validation(idx) = 1;

end

%% 8.2 Testo anomalijos

usedStarts = [];

for k = 1:numTestAnomalies

    while true

        startIdx = randi( ...
            [1, size(X_test_anomaly,1) - anomalyLength + 1]);

        if isempty(usedStarts) || ...
                all(abs(startIdx - usedStarts) >= anomalyLength)

            break;

        end

    end

    usedStarts(end+1) = startIdx;

    idx = startIdx:startIdx + anomalyLength - 1;

    % Padidiname aktyviaja galia
    X_test_anomaly(idx,1) = ...
        X_test_anomaly(idx,1) * 2.5;

    labels_test(idx) = 1;

end

% Perskaiciuojame nuo aktyvios galios priklausancias pozymes
X_validation_anomaly(:,8) = ...
    movstd(X_validation_anomaly(:,1), [3 0]);

X_validation_anomaly(:,9) = ...
    [0; diff(X_validation_anomaly(:,1))];

X_test_anomaly(:,8) = ...
    movstd(X_test_anomaly(:,1), [3 0]);

X_test_anomaly(:,9) = ...
    [0; diff(X_test_anomaly(:,1))];

fprintf('Validacijos anomaliju: %d\n', sum(labels_validation));
fprintf('Testo anomaliju: %d\n', sum(labels_test));

%% 9. BASELINE
% Moving Average + Z-score

fprintf('\n========================================\n');
fprintf('BASELINE\n');
fprintf('Moving Average + Z-score\n');
fprintf('========================================\n');

% Mokymo duomenu slenkantis vidurkis
windowSize = 16;

baselineTrainMean = movmean( ...
    X_train(:,1), [windowSize 0]);

baselineTrainResidual = ...
    X_train(:,1) - baselineTrainMean;

baselineResidualStd = ...
    std(baselineTrainResidual);

if baselineResidualStd == 0 || ...
        ~isfinite(baselineResidualStd)

    baselineResidualStd = 1;

end

% Validacija
validationMovingMean = movmean( ...
    X_validation_anomaly(:,1), [windowSize 0]);

validationResidual = ...
    X_validation_anomaly(:,1) - validationMovingMean;

validationZ = ...
    abs(validationResidual) / baselineResidualStd;

% Testas
testMovingMean = movmean( ...
    X_test_anomaly(:,1), [windowSize 0]);

testResidual = ...
    X_test_anomaly(:,1) - testMovingMean;

testZ = ...
    abs(testResidual) / baselineResidualStd;

%% 9.1 Baseline slenkscio pasirinkimas

candidateThresholds = linspace(1,5,200);

bestThreshold = NaN;
bestScore = -inf;

for i = 1:length(candidateThresholds)

    currentThreshold = candidateThresholds(i);

    metrics = calculateMetrics( ...
        validationZ, ...
        labels_validation, ...
        currentThreshold);

    if metrics.WarningsPerDay <= 5

        if metrics.Recall > bestScore

            bestScore = metrics.Recall;
            bestThreshold = currentThreshold;

        end

    end

end

% Jei nebuvo tinkamo slenkscio
if isnan(bestThreshold)

    bestThreshold = 3;

end

baselineThreshold = bestThreshold;

baselineResult = calculateMetrics( ...
    testZ, ...
    labels_test, ...
    baselineThreshold);

fprintf('Threshold = %.4f\n', baselineThreshold);
fprintf('Recall = %.4f\n', baselineResult.Recall);
fprintf('Warnings/day = %.2f\n', ...
    baselineResult.WarningsPerDay);

%% 10. Duomenu standartizavimas

meanTrain = mean(X_train,1);
stdTrain = std(X_train,[],1);

% Apsauga nuo nulio
stdTrain(stdTrain == 0) = 1;

% Apsauga nuo NaN
meanTrain(~isfinite(meanTrain)) = 0;
stdTrain(~isfinite(stdTrain)) = 1;

X_train_s = ...
    (X_train - meanTrain) ./ stdTrain;

X_validation_s = ...
    (X_validation_anomaly - meanTrain) ./ stdTrain;

X_test_s = ...
    (X_test_anomaly - meanTrain) ./ stdTrain;

%% 10.1 Galutinis NaN tikrinimas

if any(~isfinite(X_train_s(:)))

    error('X_train turi NaN arba Inf reiksmiu.');

end

if any(~isfinite(X_validation_s(:)))

    error('X_validation turi NaN arba Inf reiksmiu.');

end

if any(~isfinite(X_test_s(:)))

    error('X_test turi NaN arba Inf reiksmiu.');

end

fprintf('\nDuomenu standartizavimas baigtas.\n');
fprintf('NaN/Inf reiksmiu neliko.\n');

%% 11. ONE-CLASS SVM
% Pagrindinis metodas

fprintf('\n========================================\n');
fprintf('ONE-CLASS SVM\n');
fprintf('PAGRINDINIS METODAS\n');
fprintf('========================================\n');

maxSVMTrain = 30000;

rng(42);

if size(X_train_s,1) > maxSVMTrain

    svmIdx = randperm( ...
        size(X_train_s,1), ...
        maxSVMTrain);

    X_svm_train = X_train_s(svmIdx,:);

else

    X_svm_train = X_train_s;

end

fprintf('SVM mokymui naudojama: %d irasu\n', ...
    size(X_svm_train,1));

%% 11.1 SVM mokymas

svmModel = fitcsvm( ...
    X_svm_train, ...
    ones(size(X_svm_train,1),1), ...
    'KernelFunction', 'rbf', ...
    'KernelScale', 'auto', ...
    'OutlierFraction', 0.01, ...
    'Standardize', false);

fprintf('One-Class SVM mokymas baigtas.\n');

%% 11.2 SVM validacija

[~, svmValidationScore] = ...
    predict(svmModel, X_validation_s);

svmValidationScore = ...
    -svmValidationScore;

%% 11.3 SVM testas

[~, svmTestScore] = ...
    predict(svmModel, X_test_s);

svmTestScore = ...
    -svmTestScore;

%% 11.4 SVM slenkscio pasirinkimas

candidateThresholds = linspace( ...
    min(svmValidationScore), ...
    max(svmValidationScore), ...
    200);

bestThreshold = NaN;
bestScore = -inf;

for i = 1:length(candidateThresholds)

    currentThreshold = candidateThresholds(i);

    metrics = calculateMetrics( ...
        svmValidationScore, ...
        labels_validation, ...
        currentThreshold);

    if metrics.WarningsPerDay <= 5

        if metrics.Recall > bestScore

            bestScore = metrics.Recall;
            bestThreshold = currentThreshold;

        end

    end

end

if isnan(bestThreshold)

    bestThreshold = ...
        prctile(svmValidationScore,99);

end

svmThreshold = bestThreshold;

svmResult = calculateMetrics( ...
    svmTestScore, ...
    labels_test, ...
    svmThreshold);

fprintf('Threshold = %.4f\n', svmThreshold);
fprintf('Recall = %.4f\n', svmResult.Recall);
fprintf('Warnings/day = %.2f\n', ...
    svmResult.WarningsPerDay);

%% 12. MLP AUTOENCODER

fprintf('\n========================================\n');
fprintf('MLP AUTOENCODER\n');
fprintf('========================================\n');

fprintf('Architektura: 9-8-3-8-9\n');

inputSize = 9;
hidden1Size = 8;
hidden2Size = 3;
hidden3Size = 8;

rng(42);

% Pradiniai svoriai
W1 = randn(inputSize, hidden1Size) * 0.1;
b1 = zeros(1, hidden1Size);

W2 = randn(hidden1Size, hidden2Size) * 0.1;
b2 = zeros(1, hidden2Size);

W3 = randn(hidden2Size, hidden3Size) * 0.1;
b3 = zeros(1, hidden3Size);

W4 = randn(hidden3Size, inputSize) * 0.1;
b4 = zeros(1, inputSize);

eta = 0.01;
iterations = 30;

fprintf('Mokymo iteraciju: %d\n', iterations);
fprintf('Mokomas autoencoder...\n');

%% 12.1 Autoenkoderio mokymas

for iteration = 1:iterations

    totalError = 0;

    for n = 1:size(X_train_s,1)

        x = X_train_s(n,:);

        %% Forward propagation

        v1 = x * W1 + b1;
        y1 = 1 ./ (1 + exp(-v1));

        v2 = y1 * W2 + b2;
        y2 = 1 ./ (1 + exp(-v2));

        v3 = y2 * W3 + b3;
        y3 = 1 ./ (1 + exp(-v3));

        v4 = y3 * W4 + b4;
        y4 = v4;

        %% Klaida

        e = x - y4;

        totalError = ...
            totalError + mean(e.^2);

        %% Backpropagation

        delta4 = e;

        delta3 = ...
            (y3 .* (1-y3)) .* ...
            (delta4 * W4');

        delta2 = ...
            (y2 .* (1-y2)) .* ...
            (delta3 * W3');

        delta1 = ...
            (y1 .* (1-y1)) .* ...
            (delta2 * W2');

        %% Svoriu atnaujinimas

        W4 = ...
            W4 + eta * (y3' * delta4);

        b4 = ...
            b4 + eta * delta4;

        W3 = ...
            W3 + eta * (y2' * delta3);

        b3 = ...
            b3 + eta * delta3;

        W2 = ...
            W2 + eta * (y1' * delta2);

        b2 = ...
            b2 + eta * delta2;

        W1 = ...
            W1 + eta * (x' * delta1);

        b1 = ...
            b1 + eta * delta1;

    end

    totalError = ...
        totalError / size(X_train_s,1);

    fprintf( ...
        'Iteracija %d/%d, klaida = %.6f\n', ...
        iteration, ...
        iterations, ...
        totalError);

end

fprintf('Autoencoder mokymas baigtas.\n');

%% 12.2 Autoenkoderio validacija

validationAE = ...
    zeros(size(X_validation_s,1),1);

for n = 1:size(X_validation_s,1)

    x = X_validation_s(n,:);

    y1 = ...
        1 ./ (1 + exp(-(x * W1 + b1)));

    y2 = ...
        1 ./ (1 + exp(-(y1 * W2 + b2)));

    y3 = ...
        1 ./ (1 + exp(-(y2 * W3 + b3)));

    output = ...
        y3 * W4 + b4;

    validationAE(n) = ...
        mean((x - output).^2);

end

%% 12.3 Autoenkoderio testas

testAE = ...
    zeros(size(X_test_s,1),1);

for n = 1:size(X_test_s,1)

    x = X_test_s(n,:);

    y1 = ...
        1 ./ (1 + exp(-(x * W1 + b1)));

    y2 = ...
        1 ./ (1 + exp(-(y1 * W2 + b2)));

    y3 = ...
        1 ./ (1 + exp(-(y2 * W3 + b3)));

    output = ...
        y3 * W4 + b4;

    testAE(n) = ...
        mean((x - output).^2);

end

%% 12.4 Autoenkoderio slenkscio pasirinkimas

candidateThresholds = linspace( ...
    min(validationAE), ...
    max(validationAE), ...
    200);

bestThreshold = NaN;
bestScore = -inf;

for i = 1:length(candidateThresholds)

    currentThreshold = candidateThresholds(i);

    metrics = calculateMetrics( ...
        validationAE, ...
        labels_validation, ...
        currentThreshold);

    if metrics.WarningsPerDay <= 5

        if metrics.Recall > bestScore

            bestScore = metrics.Recall;
            bestThreshold = currentThreshold;

        end

    end

end

if isnan(bestThreshold)

    bestThreshold = ...
        prctile(validationAE,99);

end

aeThreshold = bestThreshold;

aeResult = calculateMetrics( ...
    testAE, ...
    labels_test, ...
    aeThreshold);

fprintf('Threshold = %.6f\n', aeThreshold);
fprintf('Recall = %.4f\n', aeResult.Recall);
fprintf('Warnings/day = %.2f\n', ...
    aeResult.WarningsPerDay);

%% 13. GALUTINIAI REZULTATAI

fprintf('\n========================================\n');
fprintf('GALUTINIAI REZULTATAI\n');
fprintf('========================================\n');

Method = { ...
    'Moving Average + Z-score'; ...
    'One-Class SVM'; ...
    'MLP Autoencoder'};

Recall = [ ...
    baselineResult.Recall; ...
    svmResult.Recall; ...
    aeResult.Recall];

WarningsPerDay = [ ...
    baselineResult.WarningsPerDay; ...
    svmResult.WarningsPerDay; ...
    aeResult.WarningsPerDay];

Stability = [ ...
    baselineResult.Stability; ...
    svmResult.Stability; ...
    aeResult.Stability];

resultsTable = table( ...
    Method, ...
    Recall, ...
    WarningsPerDay, ...
    Stability);

disp(resultsTable);

writetable( ...
    resultsTable, ...
    'rezultatai.csv');

fprintf('\nRezultatai issaugoti i rezultatai.csv\n');

%% 14. ABLACIJA
% SVM tik su Global Active Power

fprintf('\n========================================\n');
fprintf('ABLACIJA\n');
fprintf('========================================\n');

X_train_one = X_train(:,1);
X_validation_one = X_validation_anomaly(:,1);
X_test_one = X_test_anomaly(:,1);

oneMean = mean(X_train_one);
oneStd = std(X_train_one);

if oneStd == 0 || ~isfinite(oneStd)

    oneStd = 1;

end

X_train_one_s = ...
    (X_train_one - oneMean) / oneStd;

X_validation_one_s = ...
    (X_validation_one - oneMean) / oneStd;

X_test_one_s = ...
    (X_test_one - oneMean) / oneStd;

maxSVMTrain = 30000;

rng(42);

if size(X_train_one_s,1) > maxSVMTrain

    svmIdx = randperm( ...
        size(X_train_one_s,1), ...
        maxSVMTrain);

    X_one_train = ...
        X_train_one_s(svmIdx);

else

    X_one_train = X_train_one_s;

end

svmOneFeature = fitcsvm( ...
    X_one_train, ...
    ones(size(X_one_train,1),1), ...
    'KernelFunction', 'rbf', ...
    'KernelScale', 'auto', ...
    'OutlierFraction', 0.01, ...
    'Standardize', false);

[~, oneValidationScore] = ...
    predict(svmOneFeature, X_validation_one_s);

oneValidationScore = ...
    -oneValidationScore;

[~, oneTestScore] = ...
    predict(svmOneFeature, X_test_one_s);

oneTestScore = ...
    -oneTestScore;

candidateThresholds = linspace( ...
    min(oneValidationScore), ...
    max(oneValidationScore), ...
    200);

bestThreshold = NaN;
bestScore = -inf;

for i = 1:length(candidateThresholds)

    currentThreshold = candidateThresholds(i);

    metrics = calculateMetrics( ...
        oneValidationScore, ...
        labels_validation, ...
        currentThreshold);

    if metrics.WarningsPerDay <= 5

        if metrics.Recall > bestScore

            bestScore = metrics.Recall;
            bestThreshold = currentThreshold;

        end

    end

end

if isnan(bestThreshold)

    bestThreshold = ...
        prctile(oneValidationScore,99);

end

oneFeatureThreshold = bestThreshold;

oneFeatureResult = calculateMetrics( ...
    oneTestScore, ...
    labels_test, ...
    oneFeatureThreshold);

fprintf('SVM tik su Global Active Power:\n');
fprintf('Recall = %.4f\n', ...
    oneFeatureResult.Recall);

fprintf('Warnings/day = %.2f\n', ...
    oneFeatureResult.WarningsPerDay);

%% 15. ATSPARUMO TESTAS

fprintf('\n========================================\n');
fprintf('ATSPARUMO TESTAS\n');
fprintf('========================================\n');

noiseLevel = 0.05;

rng(100);

X_test_noisy = X_test_anomaly;

% Santykinis triuksmas aktyviajai galiai
activePowerNoise = ...
    noiseLevel * randn(size(X_test_noisy(:,1)));

X_test_noisy(:,1) = ...
    X_test_noisy(:,1) .* ...
    (1 + activePowerNoise);

% Negalime tureti neigiamos galios
X_test_noisy(:,1) = ...
    max(X_test_noisy(:,1),0);

% Perskaiciuojame priklausomas pozymes
X_test_noisy(:,8) = ...
    movstd(X_test_noisy(:,1), [3 0]);

X_test_noisy(:,9) = ...
    [0; diff(X_test_noisy(:,1))];

% Standartizavimas
X_test_noisy_s = ...
    (X_test_noisy - meanTrain) ./ stdTrain;

% SVM prognoze
[~, noisyScore] = ...
    predict(svmModel, X_test_noisy_s);

noisyScore = ...
    -noisyScore;

noisyResult = calculateMetrics( ...
    noisyScore, ...
    labels_test, ...
    svmThreshold);

fprintf('Triuksmo lygis = %.2f\n', noiseLevel);
fprintf('Recall su triuksmu = %.4f\n', ...
    noisyResult.Recall);

fprintf('Warnings/day su triuksmu = %.2f\n', ...
    noisyResult.WarningsPerDay);

%% 15.1 Sprendimo stabilumas

normalDecision = ...
    svmTestScore > svmThreshold;

noisyDecision = ...
    noisyScore > svmThreshold;

decisionChangeRate = ...
    mean(normalDecision ~= noisyDecision);

fprintf('Sprendimo pokycio dalis = %.4f\n', ...
    decisionChangeRate);

%% 16. Grafikas: SVM aptiktos anomalijos

figure;

plot( ...
    X_test_anomaly(:,1), ...
    'LineWidth', 1);

hold on;

anomalyIdx = ...
    find(svmTestScore > svmThreshold);

plot( ...
    anomalyIdx, ...
    X_test_anomaly(anomalyIdx,1), ...
    'rx');

xlabel('Laiko intervalas');
ylabel('Global Active Power');

title('SVM aptiktos anomalijos');

legend( ...
    'Aktyvioji galia', ...
    'Aptiktos anomalijos');

grid on;

%% 17. Grafikas: Recall palyginimas

figure;

bar(Recall);

set(gca, ...
    'XTickLabel', Method);

ylabel('Recall');

title('Anomaliju aptikimo Recall');

grid on;

%% 18. Grafikas: ispejimai per diena

figure;

bar(WarningsPerDay);

set(gca, ...
    'XTickLabel', Method);

ylabel('Ispejimai per diena');

title('Ispejimu skaicius per diena');

grid on;

fprintf('\n========================================\n');
fprintf('ANALIZE BAIGTA\n');
fprintf('========================================\n');


%% FUNKCIJA: METRIKOS

function result = calculateMetrics( ...
    scores, ...
    labels, ...
    threshold)

    predicted = ...
        scores > threshold;

    TP = sum( ...
        predicted == 1 & labels == 1);

    FN = sum( ...
        predicted == 0 & labels == 1);

    if (TP + FN) == 0

        recall = 0;

    else

        recall = ...
            TP / (TP + FN);

    end

    % 15 minuciu intervalai
    intervalsPerDay = 24 * 4;

    warningsPerDay = ...
        sum(predicted) / ...
        length(predicted) * ...
        intervalsPerDay;

    % Paprastas stabilumo rodiklis
    stability = ...
        1 / (1 + warningsPerDay);

    result.Recall = recall;
    result.WarningsPerDay = warningsPerDay;
    result.Stability = stability;

end
