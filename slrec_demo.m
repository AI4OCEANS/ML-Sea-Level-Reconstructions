%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Demo Sea Level Reconstructions
%
%   Description: Matlab script to reconstruct regional sea level
%   variability from proxy data using machine learning (Gaussian Processes
%   or Recurrent Neural Networks).
% 
%
%   Notes: This is an example for the North East Pacific Ocean region.
% 
% 
%   Created 07/08/2021 by Cristina Radin.
%   Last update 15/08/2021.
%
%   Background:
%   Principal Investigator: Veronica Nieves
%   The methodology used to reconstruct historical sea level records 
%   from C.Radin and V. Nieves (2021)...
%
%
%   Copyright 2021 www.aiforoceans.org 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load the dataset

load('slrec_dataset.mat')

%% Get variables

name=slrec_nepo.name;
region_title=slrec_nepo.region_title;
initial_year=slrec_nepo.initial_year;
neurons=slrec_nepo.neurons;
sl=slrec_nepo.sl;
time_sl=slrec_nepo.time_sl;
slproxy=slrec_nepo.slproxy;
time_proxy=slrec_nepo.time_proxy;
stations=slrec_nepo.stations;

%% Grab tide gauge dataset

for i=1:1%length(stations)       
id(i)=stations(i).id;
name_id(i)=stations(i).name_id;
tg(:,i)=stations(i).tg;
time_tg(:,i)=stations(i).time_tg;
end

%% Model reconstruction 

% Choose method
method= 'gp'; %'GP' or 'RNN'
preprocessing= 'yes'; % 'yes' or 'no'

tic 

RESULTS_SLREC = slrec(slproxy,tg, time_proxy, time_tg, initial_year, preprocessing, method, neurons);

toc 

%% Plot results

% Extract output variables 

time    =   RESULTS_SLREC.time;
YPred   =   RESULTS_SLREC.YPred;
Y       =   RESULTS_SLREC.Y;



if strcmp(upper(method), 'GP')
    
    intPred =   RESULTS_SLREC.intPred;
    
    figure,
    frame_h=get(handle(gcf), 'JavaFrame');
    set(frame_h,'Maximized',1);
    p1=plot(time,YPred, '-', 'Color', '#0772BF',  'LineWidth', 4,'DisplayName', 'REC (PROXY)'); hold on
    p2=plot(time,Y, '-k','LineWidth', 4, 'DisplayName', 'OBS TG');
    hold on
    ax = gca;
    ylabel('Sea Level (mm)')
    hline=refline([0 0]);
    hline.Color='k';
    hline.LineWidth = 0.2;
    hline.LineStyle = '-';
    yl=ylim(ax) + [-1,1]*range(ylim(ax)).* 0.02;
    ylim(ax, yl) %0.08
    fill([time' fliplr(time')], [intPred(:,1)', fliplr(intPred(:,2)')], [0.0275,  0.4471, 0.7490], 'FaceAlpha',0.2, 'EdgeColor','none');
    hold off
    
    get(gca,'SortMethod');
    set(gca, 'SortMethod', 'depth');
    p1.ZData = 2*ones(size(p1.XData));
    p2.ZData = ones(size(p2.XData));
    title(sprintf('%s: %s',char(region_title),char(name_id(i))))
    
    lgd=legend([p2 p1]);
    legend('boxoff');
    xlim([min(time), max(time)])
    ax.TickDir = 'in';
    ax.XMinorTick='on';
    
    xtick=[1:10*4:length(time)];
    v2=time(xtick);
    ax.XTick = v2;
    datstr=datestr((time(xtick)));
    ax.XTickLabel = datstr(:,8:end);
    
    v=1:2*4:length(time);
    ax.XAxis.MinorTickValues = time(v);
    
elseif  strcmp(upper(method), 'RNN')
    
    figure,
    frame_h=get(handle(gcf), 'JavaFrame');
    set(frame_h,'Maximized',1);
    p1=plot(time,YPred, '-', 'Color', '#0772BF',  'LineWidth', 4,'DisplayName', 'REC (PROXY)'); hold on 
    p2=plot(time,Y, '-k','LineWidth', 4, 'DisplayName', 'OBS TG');
    hold on
    ax = gca;
    ylabel('Sea Level (mm)')
    hline=refline([0 0]);
    hline.Color='k';
    hline.LineWidth = 0.2;
    hline.LineStyle = '-';
    yl=ylim(ax) + [-1,1]*range(ylim(ax)).* 0.02;
    ylim(ax, yl)
    get(gca,'SortMethod');
    set(gca, 'SortMethod', 'depth');
    p1.ZData = 2*ones(size(p1.XData));
    p2.ZData = ones(size(p2.XData));
    title(sprintf('%s: %s',char(region_title),char(name_id(i))))
    lgd=legend([p2 p1]);
    legend('boxoff');
    xlim([min(time), max(time)])
    
    
    ax.TickDir = 'in';
    ax.XMinorTick='on';
    xtick=[1:10*4:length(time)];
    v2=time(xtick);
    ax.XTick = v2;
    datstr=datestr((time(xtick)));
    ax.XTickLabel = datstr(:,8:end);    
    v=1:2*4:length(time);
    ax.XAxis.MinorTickValues = time(v);
    
end