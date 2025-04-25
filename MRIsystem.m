%% Define MRI system hardware limits
% Dependencies:
% https://github.com/pulseq/pulseq.git
% https://github.com/toppeMRI/toppe.git

% RF/gradient delay. Conservative value that should work across all GE scanners.
psd_rf_wait = 200e-6; % s

% Create Siemens/pulseq MRI system
% Here we extend rfRingdownTime by psd_rf_wait to ensure that the subsequent 
% wait pulse (delay block) doesn't overlap with the 'true' RF ringdown time
% (54 us).
sys = mr.opts('maxGrad', 40, 'gradUnit','mT/m', ...
              'maxSlew', 120, 'slewUnit', 'T/m/s', ...
              'rfDeadTime', 100e-6, ...
              'rfRingdownTime', 60e-6 + psd_rf_wait, ...
              'adcDeadTime', 20e-6, ...
              'adcRasterTime', 2e-6, ...
              'rfRasterTime', 2e-6, ...
              'gradRasterTime', 4e-6, ...
              'blockDurationRaster', 4e-6, ...
              'B0', 3.0);

% Create GE MRI system
sysGE = toppe.systemspecs('maxGrad', sys.maxGrad/sys.gamma*100, ...   % G/cm
    'maxSlew', sys.maxSlew/sys.gamma/10, ...           % G/cm/ms
    'maxRF', 0.25);

% Common raster time of Siemens: 10e-6, GE: 4e-6;
CRT = 20e-6; % s

% ADC sample time
dwell = 4e-6; % s