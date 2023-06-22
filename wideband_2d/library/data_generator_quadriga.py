# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:22:05 2023

@author: Tobias Kallehauge
"""

import numpy as np
import h5py
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from scipy.stats import multivariate_normal
import pickle
from scipy.interpolate import interp1d
from cramer_rao_lower_bound_quadriga import Cramer_Rao_Lower_Bound
import warnings


class data_generator_quadriga:
    """
    Class for loading data generated from Quadriga used for location-based
    rate selection.
    """
    def __init__(self, data_index, data_path='../Quadriga/Stored/',
                 BS_data=0,
                 l_edge=10,
                 SNR_dB=50, N_sub=600, W_MHz=20,
                 cal_fading_stats=True, cal_loc_stats = True,                 
                 eps=1e-4, oversample=100,
                 N_CDF=5_000, CRLB_monte_carlo_sims=1, **kwargs):
        """
        Initialize class and simulate fading using the channel impulse response
        from quadriga. Save the epsilon-outage capacity along with a discrete 
        sampling of the CDF for the capacity.

        The dimensions of the scenario and BS location is determined in 
        quadriga.

        Parameters
        ----------
        data_index : int
            Index for datafile in the Quadriga folder.
        data_path: str, optional
            Path to datafile. The default is '../Quadriga/Stored/'
        BS_data : int, optional
            The datafile contains channel simulations for multible BS's. The
            BS_data is the index of the BS which is used for data transmission.
            The default is 0 meaning that the UE's will make data transmissions
            with BS 0. 
        l_edge : int, optional
            Edge around the cell where the meta probability and throughput
            ratio will not be computed. l_edge controls the width of the edge.
            The default is 20.
        SNR_dB : float, optional
            The coeficients from quadriga assumes a transmit power of 1 dB, 
            we multiply with the SNR here the default is 30 dB. 
        N_sub : int, optional
            Number of subcarriers. This number should be odd since this greatly
            simplifies computation of localization statistics.            
            The default is 601
        W_MHz : float, optional
            Bandwidth in mega Hertz. 
        cal_fading_stats : bool, optional
            If True compute fading statistics. Otherwise give path of file 
            where the statistics are already computed. 
            The default is True.
        cal_loc_stats : bool, optional
            If True compute localization statistics. Otherwise give path of
            file where the statistics are already computed. 
            The default is True.
        eps: float, optional
            Target outage probability. The default is 1e-4. 
        oversample : int, optional
            When the epsilon-outage capacity is estimated non-parametrically.
            Here we sample oversample*epsilon^(-1) times of the capacity
            and then estimate the epsilon quantile from that. 
            The default is 100.
        N_CDF : int, optional
            The CDF of the capacity for each location is saved for N_CDF 
            discrete number of points. The CDF is used to compute the outage 
            probability used to compute the throughput ratio. 
        CRLB_monte_carlo_sims : int, optional
            The number of fading simulations used for numerically integrating the conditional
            CRLB. The default is 1.
        """

        # fist load configurations
        self.data_index = data_index
        self.config = pd.read_csv(data_path +
                                  f'Distribution_map_{data_index}_config.csv',
                                  index_col=0, header=None, names=['value'])
        # make function for extracing value based on name
        def get_num(name): return eval(self.config.loc[name].value)
        self.extent = [get_num('x_min'), get_num('x_max'),
                       get_num('y_min'), get_num('y_max')]
        self.N_sim = get_num('N_points_dist_map')
        self.N_side_cal = get_num('N_side')
        self.f = get_num('frec')
        self.K = get_num('NumClusters')
        self.N_BS = get_num('N_BS')

        # load data
        dat = h5py.File(data_path +
                        f'Distribution_map_{data_index}_radio_map.h5')

        # setup locations
        self.x_BS = dat['BS_coordinates'][()].T
        self.x_cal = dat['ue_coordinates'][()].T
        self.x_max = max(self.extent) - l_edge  # assume square cell centered at zero

        self.idx_eval_bool = (-self.x_max <= self.x_cal[:, 0]) & \
                             (self.x_cal[:, 0] <= self.x_max) & \
                             (-self.x_max <= self.x_cal[:, 1]) & \
                             (self.x_cal[:, 1] <= self.x_max)
        self.x_eval = self.x_cal[self.idx_eval_bool]
        self.l_edge = l_edge
        self.idx_eval = np.arange(self.N_sim)[self.idx_eval_bool]
        self.N_eval = self.x_eval.shape[0]
        self.N_side_eval = round(np.sqrt(self.N_eval))
        self.extent_eval = [-self.x_max,self.x_max,-self.x_max, self.x_max]

        # load coefficients and delays
        # size of coeff and delay are (N_sim, N_tx, N_paths)
        self.coeff = dat['coeff_real'][()].T + 1j*dat['coeff_imag'][()].T
        self.delay = dat['delay'][()].T
        self.BS_data = BS_data  # index for BS used for data transmission

        # setup channel parameters
        self.SNR = 10**(SNR_dB/10)
        if N_sub % 2 == 0:
            N_sub += 1
            warnings.warn(f'Number of subcarriers should be odd. Changed to {N_sub}')
        self.N_sub = N_sub  # number of subcarriers
        self.W = W_MHz*10**6  # bandwidth
        self.df = self.W/self.N_sub  # subcarrier spacing

        dat.close()

        # estimate fading statistics
        if cal_fading_stats:
            self.eps = eps
            self.R_eps, self.p, self.q = self.estimate_fading_stats(eps,
                                                                    oversample,
                                                                    N_CDF,
                                                                    **kwargs)
        else:
            assert 'res_path' in kwargs, 'Provide path for computed statistics'

            with open(kwargs['res_path'], 'rb') as f:
                stats = pickle.load(f)
            self.eps = eps
            self.R_eps = stats['R_eps']
            self.p = stats['p']
            self.q = stats['q']

        # get throughput of R_eps (only at x_eval)
        self.T_eps_eval = self.R_eps[self.idx_eval]*(1-self.eps)

        if cal_loc_stats:
            # estimate localization bounds
            CRLB_class = Cramer_Rao_Lower_Bound(self.x_cal,
                                                [x_BS for x_BS in self.x_BS],
                                                [self.delay[:, i, :] for i in range(self.N_BS)],
                                                [self.coeff[:, i, :] for i in range(self.N_BS)],
                                                self.SNR,
                                                self.W,
                                                self.N_sub)
            # Sigma_sim is not saved to class
            self.cov_loc_cal, Sigma_sim = CRLB_class(CRLB_monte_carlo_sims,**kwargs)  # size=(M, d, d)
            self.cov_loc_eval = self.cov_loc_cal[self.idx_eval]
            self.PEB = np.sqrt(np.trace(self.cov_loc_eval, axis1=1, axis2=2))
            

            path = kwargs.get('crlb_path',
                              f'data/CRLB_idx_{self.data_index}_SNRdB{SNR_dB}.pickle')
            res = {'cov_loc_cal': self.cov_loc_cal,
                   'cov_loc_eval': self.cov_loc_eval,
                   'Sigma_sim': Sigma_sim,
                   'PEB': self.PEB}
            with open(path, 'wb') as f:  # Overwrites any existing file.
                pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)

        else:
            assert 'crlb_path' in kwargs, 'Provide path for computed statistics'

            with open(kwargs['crlb_path'], 'rb') as f:
                CRLB_data = pickle.load(f)
            self.cov_loc_cal = CRLB_data['cov_loc_cal']
            self.cov_loc_eval = CRLB_data['cov_loc_eval']
            self.PEB = CRLB_data['PEB']
            del CRLB_data
            
        # self.cov_loc_cal = np.repeat(np.diag([10,10]), self.N_sim).reshape(self.N_sim,2,2,order = 'F')
        # self.cov_loc_eval = np.repeat(np.diag([10,10]), self.N_eval).reshape(self.N_eval,2,2,order = 'F')
      
          
        # get number of paths within 1/W of the LOS path
        self.Kprime = ((self.delay[:,:, 1:] - self.delay[:,:,0][:,:,None]) <
                       1/self.W).sum(axis = 2)

            
         

    def estimate_fading_stats(self, eps, oversample, N_CDF,
                              multiprocessing=False,
                              N_workers=4,
                              save=True,
                              **kwargs):
        """
        Estimate etimate small-scale fading statistics including epsilon-outage
        capacity and descrete CDF for capacity. Based on the channel between
        BS with index BS_data and each possible UE location on the map. 

        Parameters
        ----------
        multiprocessing : bool, optional
            If true, multiprocessing is used for simulation. 
            The default is False.
        N_workers : int, optional
            Numeber of workers if multiprocessing is used. 
        save : bool, optional
            If True save the estimated statistics as pickle file. Note that
            res_path can be provided as optional keyword argument to specify 
            where the files are saved. The default is True. 

        Optional parameters
        -------------------
        p_min : float, optional
            Minimum probability where the quantile is estimated. The default is 1e-5. 
        p_max: float, optional
            Maximum probability where the quantile is estimated. The default is 1.
        res_path : str, optional
            If save is True, the file is saved at this path. The default is 
            data/stats_idx_{self.data_index}.pickle. 

        See init method for description of the other parameters. 

        Returns
        -------
        R_eps : np.ndarray
            Epsilon outage capacity for each location in dataset. 
        p : np.ndarray
            Probabilties for discrete CDF
        q : quantiles for discrete CDf corresponding to p. 
        """

        # Estimate epsilon-outage capacity and CDF of capacity
        p_min = kwargs.get('p_min', 1e-5)
        p_max = kwargs.get('p_max', 1)
        # probabilites with logarichmic distancing
        p = 10**(np.linspace(np.log10(p_min), np.log10(p_max), N_CDF))
        q = np.zeros((self.N_sim, N_CDF))  # quantiles for probabilities
        R_eps = np.zeros(self.N_sim)  # outage capacities

        # setup simulation
        N_phase = int(oversample/self.eps)
        M = kwargs.get('M', int(1e8))  # maximum size of tensors in computation below
        # how many interations to do the simulation loop over
        N_split = max(1, N_phase*self.K*self.N_sub//M)
        # how many random samples in each loop (rounded up)
        N_phase_loop = int(np.ceil(N_phase/N_split))

        print('Estimating capacity statistics')
        if not multiprocessing:
            R_eps, q = self._estimate_stats_core(self.delay[:, self.BS_data],
                                                 self.coeff[:, self.BS_data],
                                                 eps, p, N_CDF, N_split,
                                                 N_phase_loop, self.N_sub,
                                                 self.K, self.df, self.SNR)
        else:

            # Pool with progress bar
            pool = mp.Pool(processes=N_workers,
                           initargs=(mp.RLock(),),
                           initializer=tqdm.set_lock)

            # setup arguments
            idx_split = np.array_split(np.arange(self.N_sim), N_workers)

            # run multiprocessing
            jobs = [pool.apply_async(self._estimate_stats_core,
                                     args=(self.delay[idx_split[i], self.BS_data],
                                           self.coeff[idx_split[i], self.BS_data],
                                           eps, p, N_CDF, N_split, N_phase_loop,
                                           self.N_sub, self.K, self.df, self.SNR),
                                     kwds={'worker_idx': i})
                    for i in range(N_workers)]
            pool.close()
            pool.join()

            # stack results
            for i in range(N_workers):
                job = jobs[i].get()
                R_eps[idx_split[i]] = job[0]
                q[idx_split[i]] = job[1]

        # save data if prompted
        if save:
            path = kwargs.get('res_path',
                              f'data/stats_idx_{self.data_index}.pickle')
            res = {'R_eps': R_eps, 'p': p, 'q': q}
            with open(path, 'wb') as f:  # Overwrites any existing file.
                pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)

        return(R_eps, p, q)

    def get_meta_probability(self, R, idx='all', matrix = False):
        """
        Compute meta probability given selected rate R for all locations in
        x_eval

        Parameters
        ----------
        R : np.ndarray
            Selected rate for locations in x_cal (should be same length).
            Note that meta probability is only computed at x_eval
        idx : str or np.ndarray
            Two options:
                all : evaluate meta-probability at all locations
                numpy array : Index list from x_eval where it should be 
                evaluted. 
        matrix : bool, optional
            If True, the R is a matrix where each row has selected rates for 
            different true locations. If false the rate does note depend on 
            the true location and R is just a vector. R should allways 
            corrospond to the selec indicies. The default is False. 

        Returns
        -------
        p : np.ndarray
            Meta probability for each location in x_eval
        """
        if isinstance(idx, str):
            assert idx == 'all', 'Unknown option for index!'
            idx = np.arange(self.N_eval)

        p = np.zeros(len(idx))

        A = (self.x_cal[1, 0] - self.x_cal[0, 0])**2  # area of grid lattice
        for i, j in enumerate(idx):
            idx_cal = self.idx_eval[j]
            
            if matrix:
                R_i = R[i]
            else:
                R_i = R
            
            S_idx = self.R_eps[idx_cal] < R_i  # Boolean outage region

            # the height dimension is not used below
            S = self.x_cal[S_idx][:, :2]  # locations where outage is violated

            f = multivariate_normal.pdf(S,
                                        mean=self.x_cal[idx_cal, :2],
                                        cov=self.cov_loc_eval[j])
            p[i] = A*f.sum()

        return(p)

    def get_throughput(self, R, matrix = False):
        T_ratio = np.zeros(self.N_eval)

        A = (self.x_cal[1, 0] - self.x_cal[0, 0])**2  # area of grid lattice
        for i, x in enumerate(self.x_eval):
            idx = self.idx_eval[i]
            if matrix:
                R_i = R[i]
            else:
                R_i = R

            # get CCDF for the capacity (using linear interpolation)
            CCDF = interp1d(self.q[idx], 1 - self.p,
                            bounds_error=False,
                            fill_value=(1, 0))  # 1 and 0 outside bounds

            # get  1 - p_out(R) using CCDF
            
            c_p_out = CCDF(R_i)

            # get pdf (the height dimension is not used below)
            f = multivariate_normal.pdf(self.x_cal[:, :2],
                                        mean=x[:2],
                                        cov=self.cov_loc_eval[i])

            T = A*(R_i*f*c_p_out).sum()

            T_ratio[i] = T/self.T_eps_eval[i]

        return(T_ratio)

    def save(self, path):
        """
        Saves object as a pickle file at specified path

        Parameters
        ----------
        path : str
            Path fo be saved at including name and extension
            (should be .pickle)
        """

        with open(path, 'wb') as f:  # Overwrites any existing file.
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod  # to static method does not copy class which saves RAM in multiprocessing
    def _estimate_stats_core(delay, coeff, eps, p, N_CDF, N_split,
                             N_phase_loop, N_sub, K, df, SNR,
                             worker_idx=0):
        """
        Executes estiamtion of parameters on the given indicies idx. See 
        estimate_fading_stats for further discription.    
        """

        N_core = len(delay)
        R_eps = np.zeros(N_core)
        q = np.zeros((N_core, N_CDF))

        # setup progress bar
        tqdm_text = "#" + "{}".format(worker_idx).zfill(3)
        progress_bar = tqdm(total=N_core, desc=tqdm_text,
                            position=worker_idx, ascii=True,
                            mininterval=1)

        with progress_bar as pbar:
            for i in range(N_core):
                # first get delay phase response
                d = np.exp(1j*np.pi*2*np.arange(N_sub)[None, :] *
                           df*delay[i][:, None])

                C = np.zeros(N_split*N_phase_loop)
                for j in range(N_split):
                    phases = np.random.uniform(-np.pi, np.pi, (N_phase_loop, K))

                    # get channel coefficients
                    # dimensions are (N_phase_loop, K, N_sub)
                    h = coeff[i][None, :, None]*np.exp(1j*phases[:, :, None])*d[None, :, :]

                    # sum over paths to get subcarrier values (baseband) for each simulation
                    h_sub = h.sum(axis=1)

                    # compute capacity from h_sub
                    C[j*N_phase_loop: (j + 1)*N_phase_loop] = np.log2(1 +
                                                                      SNR*np.abs(h_sub)**2).sum(axis=1)

                # get outage capacity from simulations
                R_eps[i] = np.quantile(C, eps)

                # get quantiles from simulation
                q[i] = np.quantile(C, p)

                pbar.update(1)

        return(R_eps, q)
