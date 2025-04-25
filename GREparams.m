%% Scan parameters for gradient echo (GRE) sequence. Used for sensitivity maps

% Define scanner
run('MRIsystem.m')

% Spatial parameters
res_gre = [2, 2, 2]*1e-3; % resolution (m)
fov_gre = [21.6, 21.6, 21.6]*1e-2; % field of view (m)
N_gre = round(fov_gre./res_gre); % acquisiton tensor size
Nx_gre = N_gre(1); Ny_gre = N_gre(2); Nz_gre = N_gre(3);

% Temporal parameters
NdummyZloops = 4; % number of dummy excitations to reach steady state

% Other acquisition params
fatChemShift = 3.5e-6; % 3.5 ppm
fatOffresFreq = sys.gamma*sys.B0*fatChemShift; % Hz
TE = 1/fatOffresFreq + 2e-4; % fat and water in phase for both echoes
TR = 6e-3; % constant TR
T1 = 1500e-3; % approximate T1
alpha = 180/pi*acos(exp(-TR/T1)); % flip angle (degrees)

% Sequence parameters
rfDur = 0.4e-3;  % RF duration
rf_phase_0 = 117; % RF spoiling initial phase (degrees)
nCyclesSpoil = 2; % number of spoiler cycles
Tpre = 1.0e-3; % prephasing trapezoid duration