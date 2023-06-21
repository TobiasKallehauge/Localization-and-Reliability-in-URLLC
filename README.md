# Statistical Relations between Wireless Ultra-Reliability and Location Estimation
This repository contains the code for the simulations used in the article "Statistical Relations between Wireless Ultra-Reliability and Location Estimation," which is currently under review for IEEE Transactions on Wireless Communications.

## Content
 - `quadriga/`: Files used simulate impulse responses for Sec IV. The simulations require that QuaDriGa  is "installed" in the same folder under a folder called "Lisenced". Go to https://quadriga-channel-model.de/ to download QuaDriga freely.
   - `Channel.m`: Class for generating the channels within the cell as illustrated in Fig. 4.   
   - `simulate_channels.m`: Calls the class `Channel.m` and saves generated data in h5 files.
   - `Stored`: Folder containing the simulated channel coefficients and config files for five datasets used in the paper. The datasets are indexed 6-10. Dataset 6 is the one shown in most figures except Fig. 12 which contains aggregated data from all datasets. 

