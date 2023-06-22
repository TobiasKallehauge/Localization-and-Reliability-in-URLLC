# -*- coding: utf-8 -*-
"""
Created on Tue May  2 14:18:10 2023

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - Implemented the computation of the equivalent Fisher information
           for each BS position and the Fisher information
           for the position. Also implemented computation of the
           position error bound. (04/01/2022)
    v1.1 - Implemented adaptive time unit. (23/02/2023)
    v1.2 - Definition of cramer_rao_lower_bound_quadriga.py as an adaptation
           from the existing class. In this update, the input to the class
           is changed. (02/05/2023)
    v1.3 - Include clock bias. Handling numerical instabilities. (10/05/2023)
"""


import numpy as np

from fisher_information_quadriga import Fisher_Information
import multiprocessing as mp
from tqdm import tqdm


class Cramer_Rao_Lower_Bound(object):
    """
    Computes the Cramer-Rao lower bound for the UE positions. Also computes
    the position error bound (PEB).

    Methods
    --------
        __init__ : Initialize settings
        position_FIM : Compute the Fisher information matrix of the position.
        __call__ : Computes the Cramér-Rao lower bound.
    """
    def __init__(self, x_UE, x_BS, tau_list, alpha_list, SNR, W, N, c=299792458, time_unit=1e03):
        """
        Initializes the class attributes.

        Inputs:
        -------
        x_UE : ndarray, size=(M, d)
            The M UE positions in a d-dimensional space.
        x_BS : list, len=Q
            List of ndarrays of size=(d,) giving the BS positions.
        tau_list : list, len=Q
            List of ndarrays of size=(M, K) giving the delays in each UE
            position for each multipath.
        alpha_list : list, len=Q
            List of ndarrays of size=(M, K) giving the channel coefficients in
            each UE position for each multipath.
        SNR : float
            The transmit signal-to-noise-ratio.
        W : float
            The bandwidth.
        N : int
            The number of sub-carriers (minus 1).
        """
        super(Cramer_Rao_Lower_Bound, self).__init__()

        ### Assign attributes from input ###
        self.x_UE = x_UE
        self.x_BS = x_BS
        self.tau_list = tau_list
        self.alpha_list = alpha_list
        self.SNR = SNR
        self.W = W
        self.N = N
        self.c = c
        self.time_unit = time_unit

        self.d = self.x_UE.shape[1]

    def position_CRLB(self, inst_alpha_list):
        """
        Compute the Fisher information matrix of the position
        for each grid point for each BS taking clock bias into account.
        Then computes the Cramer-Rao lower bound for the position coordinates.
        """
        FIM_classes = [Fisher_Information(self.tau_list[i], inst_alpha_list[i],
                self.SNR, self.W, self.N, time_unit=self.time_unit) for i in range(len(self.x_BS))]
        Kprime = np.array([FIM_mod.Kprime for FIM_mod in FIM_classes])
        EFIM_lists = [FIM_mod() for FIM_mod in FIM_classes]
        EFIM_array = np.array(EFIM_lists)
        EFIM_mat_array = np.zeros((EFIM_array.shape[0], *EFIM_array.shape))
        EFIM_mat_array[np.arange(EFIM_array.shape[0]), np.arange(EFIM_array.shape[0]), :] = EFIM_array
        EFIM_mat_array = EFIM_mat_array.T

        differences = [self.x_UE - x_BS for x_BS in self.x_BS]
        distances = [np.linalg.norm(diff, axis = 1) for diff in differences]
        T_mat_list = [diff/(self.c/self.time_unit*dist[:, None]) for diff, dist in zip(differences, distances)]
        T_array = np.moveaxis(np.array(T_mat_list), 0, -1)[:, :2, :]
        clock_bias_derivative = np.ones((self.x_UE.shape[0], 1, len(self.x_BS)))*self.time_unit/1e09
        T_array = np.concatenate((T_array, clock_bias_derivative), axis=1)

        JxB = np.einsum("pki,pil->pkl", T_array, np.einsum("pij,plj->pil", EFIM_mat_array, T_array))
        Jx = JxB[:, :2, :2] - (1/JxB[:, 2, 2])[:, None, None] * np.einsum("pi,pj->pij", JxB[:, :2, 2], JxB[:, 2, :2])
        CRLB = np.linalg.inv(Jx)
        for i in range(self.x_UE.shape[0]):
            # Handle non-localizable cases -> Set CRLB to NaNs
            if np.sum(np.diag(EFIM_mat_array[i, :, :]) != 0) < 3:
                CRLB[i, :, :] = None
        return CRLB, Kprime

    def __call__(self, monte_carlo_sims, multiprocessing = False,
                 N_workers = 1, **kwargs):
        """
        Main class call. Runs the Fisher analysis and computes the conditional
        Cramér-Rao lower bound for each of the UE positions for a number of
        Monte carlo simulations. Then returns the average over the Monte
        carlo runs.

        Inputs:
        -------
        monte_carlo_sims : int
            The number of Monte carlo simulations.
        multiprocessing : bool, optional
            If true, multiprocessing is used for simulation. 
            The default is False.
        N_workers : int, optional
            Numeber of workers if multiprocessing is used. 

        Returns
        -------
        avg_CRLB : ndarray, size=(M, d, d)
            The average Cramér-Rao lower bound.
        Kprime : ndarray, size=(Q, M)
            The number of multipaths in each UE location
        """
        print("\n\nEstimating Localization statistics\n")
       
        
        if not multiprocessing:
            CRLB_monte_carlo = self._simulate_core(monte_carlo_sims, worker_idx = 0)
        else:
            
            # number of tasks per core
            N_monte_core = int(np.ceil(monte_carlo_sims / N_workers)) 
            
            # Pool with progress bar
            pool = mp.Pool(processes=N_workers,
                           initargs=(mp.RLock(),),
                           initializer=tqdm.set_lock)


            # run multiprocessing
            jobs = [pool.apply_async(self._simulate_core,
                                     args=(N_monte_core,),
                                     kwds={'worker_idx': i})
                    for i in range(N_workers)]
            pool.close()
            pool.join()
            
            # stack results
            CRLB_monte_carlo = np.vstack([job.get() for job in jobs])

        avg_CRLB = np.nanmean(CRLB_monte_carlo, axis=0)
        return avg_CRLB, CRLB_monte_carlo
            
    
    def _simulate_core(self,N_monte_core, worker_idx = 0):
        """
        Core function to run specified number of Monte Carlo simulations. 
        """
        
        tqdm_text = "#" + "{}".format(worker_idx).zfill(3)
        progress_bar = tqdm(total=N_monte_core, desc=tqdm_text,
                            position=worker_idx, ascii=True,
                            mininterval=1)

        CRLB_monte_carlo = np.zeros((N_monte_core, self.x_UE.shape[0], 2, 2))
        with progress_bar as pbar:
            for i in range(N_monte_core):
                random_phases = np.random.uniform(0, 2*np.pi, size=np.array(self.alpha_list).shape)
                inst_alpha_list = [self.alpha_list[i]*np.exp(1j*random_phases[i, :, :]) for i in range(len(self.x_BS))]
                CRLB_monte_carlo[i, :, :, :], _ = self.position_CRLB(inst_alpha_list)
            
                pbar.update(1)
            
        
        return CRLB_monte_carlo



