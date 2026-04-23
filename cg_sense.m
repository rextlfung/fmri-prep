%% CG-SENSE recon with BART
% Load data
datdir = '/mnt/storage/rexfung/20260409tap/recon/';
fn_epi = strcat(datdir, 'caipi_epi_zf.mat');
fn_gre = strcat(datdir, 'gre.mat');
SENSEmethod = 'bart';
fn_smaps = strcat(datdir, 'smaps_', SENSEmethod, '.mat');
lamb = 0.0;
bart_cmd = sprintf('pics -l1 -r%f -e', lamb);
fn_recon = strcat(fn_epi(1:end-11), sprintf('_recon_cgs_l1_r%f.nii', lamb));

kdata = matfile(fn_epi); % ksp_zf

run('./params.m'); Nvcoils = 18;

%% Get sensitivity maps with either BART or PISCO
if exist(fn_smaps, 'file')
    load(fn_smaps); % smaps
else
    load(fn_gre);

    fprintf('Estimating sensitivity maps from GRE data via %s...\n', SENSEmethod)

    % Compute sensitivity maps
    tic
        [smaps_raw, emaps] = makeSmaps(ksp_gre, SENSEmethod);
    toc

    % Save for next time
    save(fn_smaps, 'smaps_raw', 'emaps', '-v7.3');
end

%% Crop and resize smaps (assuming matched origin)
smaps = smaps_raw;

% Support mask created from the last eigenvalues of the G matrices 
threshold_mask = 1;
eig_mask = zeros(Nx_gre, Ny_gre, Nz_gre);
eig_mask(find(emaps(:,:,:,end) < threshold_mask)) = 1;
smaps = smaps .* eig_mask;

% Crop in z to match EPI FoV
z_start = round((fov_gre(3) - fov(3))/fov_gre(3)/2*Nz_gre + 1);
z_end = round(Nz_gre - (fov_gre(3) - fov(3))/fov_gre(3)/2*Nz_gre);
smaps = smaps(:,:,z_start:z_end,:);

% Interpolate to match EPI data dimensions
smaps_new = zeros(Nx,Ny,Nz,Nvcoils);
for coil = 1:Nvcoils
    smaps_new(:,:,:,coil) = imresize3(smaps(:,:,:,coil),[Nx,Ny,Nz]);
end
smaps = smaps_new; clear smaps_new;

% Align x-direction of smaps with EPI data (sometimes necessary)
% smaps = flip(smaps,1);

%% Recon with CG-SENSE
tic;
img = zeros(Nx,Ny,Nz,Nframes);
parfor frame = 1:Nframes
    fprintf('Reconstructing frame %d\n', round(frame));
    data = squeeze(kdata.ksp_epi_zf(:,:,:,:,frame));
    img(:,:,:,frame) = bart(bart_cmd, data, smaps);
end
toc;

%% Write to NIfTI
niftiwrite(abs(img), fn_recon)

%% Viz
interactive4D(abs(img))