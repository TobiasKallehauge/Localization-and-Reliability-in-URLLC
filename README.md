# Statistical Relations between Wireless Ultra-Reliability and Location Estimation
This repository contains the code for the simulations used in the article "Statistical Relations between Wireless Ultra-Reliability and Location Estimation," which is currently under review for IEEE Transactions on Wireless Communications.

## Content
 - `quadriga/`: Files used to simulate impulse responses for Sec IV. The simulations require that QuaDriGa  is "installed" in the same folder under a folder called "Lisenced". Go to https://quadriga-channel-model.de/ to download QuaDriga for free.
   - `Channel.m`: Class for generating the channels within the cell as illustrated in Fig. 4.   
   - `simulate_channels.m`: Calls the class `Channel.m` and saves generated data in h5 files.
   - `Stored`: Folder containing the simulated channel coefficients and config files for five datasets used in the paper. The datasets are indexed 6-10. Dataset 6 is the one shown in most figures except Fig. 12 which contains aggregated data from all datasets.
 - `rayleigh_1d/`: Files used for the simulations in Sec. III.
   - `fig1.py`: Generates Fig. 1.
   - `fig2.py`: Generates Fig. 2.
   - `throughput_result.npy`: Data used to generate Fig. 2 (generated and then saved by `fig2.py`.
   - `plots/`: Folder with plots in pdf format.
   - `UE.png`: Graphic used to produce Fig. 1
   - `BS.png`: Graphic used to produce Fig. 1
 - `wideband_2d/` Files used for the simulations in Sec. IV.
   - `library/': Folder with the Python classes used to generate the results.
     - `data_generator_quadriga.py`: Defines a model object used for most of the scripts. Generate channel statistics based on the impulse responses generated by QuaDriGa. It can also compute meta probability and throughput ratio given the selected rate.
     - `fisher_information_quadriga.py`: Computes the Cramér Rao lower bound for TDA localization derived in App. A given specific channel coefficients
     - `cramer_rao_lower_bound_quadriga.py`: Computes average Cramér Rao lower bound over random channel coefficients.
   - `make_data.py`: Uses the `data_generator_quadriga.py` to generate channel statistics for datasets generated with quadriga, i.e., $\epsilon$-outage capacity and Cramér Rao Lower bound. Saves an instance of the class `data_generator_quadriga.py` as a pickle file, which is then used to generate results. 
   - `constant_loc_err/`: Folder with results under constant localization error in sections III.D and III.E
     - `data/`: Folder with computed statistics and results
     - `make_data.py`: Loads data generated with `make_data.py` in the folder above, replace the localization error with a constant one, and then saves the model again.
     - `make_data_different_peb.py`: Similar to `make_data.py` but creates more than one dataset, each with a different localization error (i.e., position error bound (PEB)) (used to make Fig. 13-14)
     - `rate_select_backoff.py`: Select the rate using the backoff approach and then computes the resulting meta-probability and throughput ratio. 
     - `rate_select_backoff_different_peb.py`: Select the rate using the backoff approach and then computes the resulting meta-probability and throughput ratio. This script creates more than one set of results, each with a different localization error  (used to make Fig. 13-14)
     - `rate_select_interval.py`: Select the rate using the interval approach and then computes the resulting meta-probability and throughput ratio. 
     - `rate_select_intervaæ_different_peb.py`: Select the rate using the interval approach and then computes the resulting meta-probability and throughput ratio. This script creates more than one set of results, each with a different localization error  (used to make Fig. 13-14)
     - `manual_find_peaks.py`: GUI for manually selecting locations of peaks and valleys and saving results.
     - `fig8.py`: Generates Fig. 8.
     - `fig9.py`: Generates Fig. 9.
     - `fig10.py`: Generates Fig. 10.
     - `fig11.py`: Generates Fig. 11.
     - `fig12.py`: Generates Fig. 12.
     - `fig13.py`: Generates Fig. 13.
     - `fig14.py`: Generates Fig. 14.
   - `crlb_loc_err/`: Folder with the results under localization error based on Cramér Rao Lower bound in section III.F
     - `data/`: Folder with computed statistics and results
     - `rate_select_backoff.py`: Selects rate using the backoff approach.
     - `rate_select_interval.py`: Selects rate using the interval approach.
     - `rate_select_distance.py`: Selects rate using the distance approach.
     - `fig_5_6.py`: Generates Fig. 5 and 6.
     - `fig7.py`: Generates Fig. 7.
     - `fig15.py`: Generates Fig. 15.
     - `fig16.py`: Generates Fig. 16.
       
## Dependencies
The simulations are made with Python version `3.10.9`. See requirements.txt for required packages. 
