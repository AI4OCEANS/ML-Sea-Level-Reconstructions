function RESULTS_SLREC = slrec(X,Y, time_pred, time_resp, initial_year,...
    pre_proc, analysis, neurons)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Usage: RESULTS_SLREC = slrec(X,Y, time_pred, time_resp, initial_year,
%                               pre_proc, analysis, neurons)
%
%   DESCRIPTION: Matlab script to reconstruct regional sea level
%   variability from input data (predictor)* using machine learning 
%   (Gaussian Processes, GP, or Recurrent Neural Networks, RNN). 
%   Adjustment of the time series is automatically performed. 
%   A ratio of 90:10 has been set for training and validating the models. 
%   
%   INPUTS:
%
%   X             : Vector containing the predictor for each time step 
%                   (or matrix in a multivariate case**).
%   Y             : Vector containing the response (dataset 
%                   to be reconstructed) for each time step.
%   time_pred     : Date of data expressed as a date number 
%                   (given by the predictor data).
%   time_resp     : Date of data expressed as a date number
%                   (obtained from PSMSL tide gauge data file).
%   initial_year  : Initial year of the reconstruction.
%   pre_proc      : Tide gauge data preprocessing 
%                   (detrending, 1-year smoothing) 'yes' or 'no'.
%	analysis      : 'GP' or 'RNN'.
%   neurons:      : Number of hidden units.


%   OUTPUT STRUCTURE:
%
%   X             : Vector containing the preprocessed predictor for each
%                   time step.
%   Y             : Vector containing the preprocessed response for each
%                   time step.
%   time          : Time of adjusted datasets expressed as a Matlab date
%                   number.
%   YPred         : Vector containing the prediction (the reconstructed 
%                   dataset) for each time step.
%   intPred       : 95% prediciton intervals (for the GP method).
%
%
%   Notes:
%   *  For example, regional proxy estimates.
%   ** Matrix dimensions: observations x predictors.
%
%
%   Created 07/08/2021 by Cristina Radin (cristina.radin@uv.es).
%   Last update 19/11/2021.
%
%   Background:
%   Principal Investigator: Veronica Nieves (veronica.nieves@uv.es)
%   The methodology used to reconstruct historical sea level records
%   from C.Radin and V. Nieves (2021), Geophys. Res. Letters, 
%   https://doi.org/10.1029/2021GL095382
%
%   Copyright 2021 www.aiforoceans.org
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Time adjustment between time_pred and time_resp

[m n]=size(X);

if length(time_pred)~=length(time_resp)
    time_monthly=datenum(datetime(initial_year,1,1):calmonths(1):...
        datetime(2018,12,31)');
    
    initial_day=datenum(sprintf('01-01-%d',initial_year),'dd-mm-yyyy');
    end_day=datenum('31-12-2018','dd-mm-yyyy');
    
    index_missing_beg=find(time_monthly<min(time_resp));
    index_missing_end=find(time_monthly>max(time_resp));
        
    index_time_res=find(time_resp>=initial_day & time_resp<=end_day);
    Y=Y(index_time_res);
    time_resp=time_resp(index_time_res);
    
    index_time_pred=find(time_pred>=initial_day & time_pred<=end_day);
    X=X(index_time_pred);
    time_pred=time_pred(index_time_pred);
    
    Y=[NaN(1, length(index_missing_beg)-1)'; Y; NaN(1, ...
        length(index_missing_end))'];
    
    time_step=round((time_pred(2)-time_pred(1))/30);
    
    Y=nanmean(reshape(Y', [time_step, size(X,1)]), 1);
    Y=Y';
    
else
    
    time_diff=sum(time_pred-time_resp);
    time_step=round((time_pred(2)-time_pred(1))/30);
    
    if time_diff~=0
        
        time_monthly=datenum(datetime(initial_year,1,1):calmonths(1):...
            datetime(2018,12,31)');
        initial_day=datenum(sprintf('01-01-%d',initial_year),'dd-mm-yyyy');
        end_day=datenum('31-12-2018','dd-mm-yyyy');
        
        index_missing_beg=find(time_monthly<min(time_resp));
        index_missing_end=find(time_monthly>max(time_resp));       
        
        index_time_res=find(time_resp>=initial_day & time_resp<=end_day);
        Y=Y(index_time_res);
        time_resp=time_resp(index_time_res);
              
        index_time_pred=find(time_pred>=initial_day & time_pred<=end_day);
        X=X(index_time_pred);
        time_pred=time_pred(index_time_pred);
        
        Y=[NaN(1, length(index_missing_beg)-1), Y, NaN(1,  ...
            length(index_missing_end))];
                
        time_step=round((time_pred(2)-time_pred(1))/30);
        
        Y=nanmean(reshape(Y', [time_step, size(X,1)]), 1);
        Y=Y';
        
    end   
end

%% Preprocessing: detrending and 1-year moving average

if strcmp(lower(pre_proc), 'yes')
     
    Y=detrend(Y,'omitnan');
    
    nan=sum(isnan(Y),2);
    nan_position=isnan(Y);
    
    Y=movmean(Y, 1*(12/time_step), 'omitnan');
    
    Y(nan_position)=NaN;
    
end

%% Machine learning analysis

switch upper(analysis)

    case {'GP'}
        disp('Running GP method...')
        model = fitrgp(X,Y','KernelFunction','exponential', 'FitMethod',...
            'none', 'PredictMethod', 'exact');
        [YPred, ~, intGP] = predict(model,X);
        
        RESULTS_SLREC.intPred=intGP;

    case {'RNN'}
        disp('Running RNN method...')
        test=find(isnan(Y));
        train=find(~isnan(Y));
        
        Y_train=Y(train);
        Y_test=Y(test);
        X_test=X(test,:);
        X_train=X(train,:);
        
        mu = nanmean(Y_train,1);
        sig =std(Y_train, 'omitnan');
        
        Y_train = (Y_train - mu) / sig;
        Y_test = (Y_test - mu) / sig;
        
        mu_x = nanmean(X_train,1);
        sig_x =std(X_train, 'omitnan');
        
        X_train = (X_train - mu_x) ./ sig_x;
        X_norm = (X - mu_x) ./ sig_x;
        
        val=0.1;
        
        numResponses=1;
        numFeatures=n;
        
        layers = [
            sequenceInputLayer(numFeatures,"Name","sequence")
            gruLayer(neurons,"Name","gru")
            dropoutLayer(0.3,"Name","dropout")
            fullyConnectedLayer(numResponses,"Name","fc")
            regressionLayer("Name","regressionoutput")];
        
        options = trainingOptions('adam', ...
            'MaxEpochs',3000,...
            'ValidationData',{X_train(1:round(val*length(X_train)),:)',...
            Y_train(1:round(val*length(X_train)))'}, ...
            'ValidationFrequency',2, 'ValidationPatience',10,...
            'Verbose',false);
        
        net = trainNetwork(X_train(round(val*length(X_train))+1:end,:)',...
            Y_train(round(val*length(X_train))+1:end)',layers,options);
        
        YPred = predict(net,X_norm');
        YPred = sig*YPred + mu;
        YPred = YPred;
    otherwise
        disp('Unknown learning paradigm.')
end

time=datenum(datetime(initial_year,1,1):calmonths(time_step):...
    datetime(2018,12,31));

RESULTS_SLREC.time  =    time';
RESULTS_SLREC.YPred =    YPred;
RESULTS_SLREC.Y     =    Y;
RESULTS_SLREC.X     =    X;

