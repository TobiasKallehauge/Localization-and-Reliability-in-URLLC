%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate a map of distributions. Given a scenario size and a
% position for the Tx, computes the geometric channel using Quadriga
% simulator in a mesh of points. Then, adds realizations of fast fading on
% top of the geometric channel for each point in the mesh. The script can
% save the channel object in a .mat file (for reproducibility), as well as
% the geometric channel map and the distribution map in h5 dataset format. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%------------------------------------------------------------------------
% Initialization
%------------------------------------------------------------------------
close all
clc
clear 

%------------------------------------------------------------------------
% Parameters
%------------------------------------------------------------------------

% Coordinates of the transmitter (x, y, z) in meters
x_max = 50; 
edge = 20; 
xyz_tx = [[-x_max,x_max,10];
          [x_max,x_max,10];
          [-x_max,-x_max,10];
          [x_max,-x_max,10]]; 
N_tx = size(xyz_tx,1); 

% Coordinates of the receivers (x, y, z) in meters. NOTE: leave empty for
% uniformly sampled map according to sample_distance

% Scenario size array [x_min ; x_max; y_min; y_max] in meters
size_s = [-x_max - edge; x_max + edge; -x_max - edge; x_max + edge];

% Users height in meters (for simplicity, it is assumed the same for all)
% ONLY if xyz_rx is an empty matrix
user_z = 1.5;

N_sim = 1e4; % Number of points on the map
N_side = round(sqrt(N_sim)); 
if (N_sim ~= N_side^2) 
    fprintf('Number of simulations is not a squrare number. Rounded to %d\n', N_side^2); 
end 
N_sim = N_side^2; 

% Frequency of operation      
frequency = 3.6e9;

% Channel type 
type = '3GPP_3D_UMi_LOS'; % see pg 81 of Quadriga documentation

% Fast fading correlation distance [m]
SC_lambda = 5; 
    
% Number of active clusters
NumClusters = 10; 

% Name of the dataset (it is the same for the object and the datasets)
name_index = 10;  
filename = sprintf('./Stored/Distribution_map_%d',name_index);

% For reproducibility
rng(name_index);


%------------------------------------------------------------------------
% Geometric Channel Generation
%------------------------------------------------------------------------

%Coordinates of the users (in case that xyz_rx is an empty matrix)
[XX,YY] = ndgrid(linspace(size_s(1), size_s(2), N_side),...
                 linspace(size_s(3), size_s(4), N_side));
xyz_rx = [XX(:), YY(:), user_z*ones(numel(XX),1)];


% Create channel object
ch = Channel(xyz_tx, xyz_rx, frequency, type, false, size_s, ...
               1, SC_lambda, NumClusters);
           
% Calculate geometric channel + power map if draw_power_map == true. 
% Geometric channel is only computed at positions indicated by xyz_rx.
% Power map is generated in the whole scenario size_s at sample_distance
ch.draw();

% ------------------------------------------------------------------------
% Store configuration file in .csv (File 1)
% ------------------------------------------------------------------------
labels = {'scenario'; 'x_min'; 'x_max'; 'y_min'; 'y_max'; 'N_points_dist_map'; ...
          'N_side'; 'SC_lambda'; 'NumClusters'; 'N_BS'; 'frec';};
values = [string(type); size_s(1); size_s(2); size_s(3); size_s(4);...
          N_sim; N_side; SC_lambda; NumClusters; N_tx ; frequency;];
      
 % Create directory if it does not exist
 if ~exist('Stored', 'dir')
     mkdir('Stored')
 end
 
 % Create and save config file
 writetable(table(labels,values),...
     [filename '_config.csv'],'WriteVariableNames',0); 


%------------------------------------------------------------------------
% Store channel map in .hdf5 (File 2)
%------------------------------------------------------------------------
%Remove old dataset
if exist([filename '_radio_map.h5'],'file')
    delete([filename '_radio_map.h5']);
end

%--------------------------------------------------------------------------
% Save coordinates in h5 file
%--------------------------------------------------------------------------

h5create([filename '_radio_map.h5'],'/BS_coordinates', size(xyz_tx));
h5write([filename '_radio_map.h5'],'/BS_coordinates', xyz_tx);
h5create([filename '_radio_map.h5'],'/ue_coordinates', size(xyz_rx));
h5write([filename '_radio_map.h5'],'/ue_coordinates', xyz_rx);

%--------------------------------------------------------------------------
% Save channel coefficients and delays in same h5 file
%--------------------------------------------------------------------------

% put all coefficients and delays in tensors with dimensions N_rx, N_tx,
% N_paths
coeff = zeros(N_sim, N_tx, NumClusters); 
delay = zeros(N_sim, N_tx, NumClusters); 
for i = 1:N_tx
    for j = 1:N_sim
        idx = (i-1)*N_sim + j; 
        coeff(j,i,:) = ch.H(1,idx).coeff; 
        delay(j,i,:) = ch.H(1,idx).delay; 
    end 
end 

% save coef (real then imaginary)
h5create([filename '_radio_map.h5'],'/coeff_real', [N_sim, N_tx, NumClusters]);
h5write([filename '_radio_map.h5'],'/coeff_real', real(coeff));
h5create([filename '_radio_map.h5'],'/coeff_imag', [N_sim, N_tx, NumClusters]);
h5write([filename '_radio_map.h5'],'/coeff_imag', imag(coeff));


% save delay
h5create([filename '_radio_map.h5'],'/delay', [N_sim, N_tx, NumClusters]);
h5write([filename '_radio_map.h5'],'/delay', delay);











