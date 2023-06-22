# -*- coding: utf-8 -*-
"""
Created on Tue May 2 14:15:31 2023

@author: Martin Voigt Vejling
Email: mvv@es.aau.dk

Track changes:
    v1.0 - Implemented the computation of the Fisher information matrix
           without using simplified expressions. (12/11/2022)
    v1.1 - End of hackathon. Changes to __init__() herein simulation of
           small scale fading. New delay_steering_vector method.
           Paralellisation of computations in Fisher_not_simplified().
           Moreover, introduced the method Fisher_Information_Matrix() which
           computes the Fisher information matrix using the simplified
           expressions. (04/01/2023)
    v1.2 - Implemented adaptive time unit. (23/02/2023)
    v1.3 - Definition of fisher_information_quadriga.py as an adaptation
           from the existing class. In this update, the input to the class
           is changed and the inheritance with the channel.py is
           removed. (02/05/2023)
"""


import numpy as np


class Fisher_Information(object):
    """
    Compute the Fisher information matrix, and the equivalent Fisher information
    matrix for the channel model in the inheritance class "channel".

    Methods
    -------
        __init__ : Initialize settings
        dn_tau : Steering vector.
        Fisher_information_matrix : Compute the Fisher information matrix using simplified expressions.
        Equivalent_Fisher_information : Compute the equivalent Fisher information for the LoS delay.
        __call__ : Runs the Fisher analysis and outputs the EFIM.

    Attributes:
    -----------
        tau : Delays.
        alpha : Channel coefficients.
        SNR : Signal-to-noise-ratio.
        W : Bandwidth.
        N : Number of subcarriers (minus 1).
        M : Number of UE positions.
        K : Number of multipaths.
        delta_f : Frequency spacing.
        Kprime : Number of non-resolvable multipaths.
    """
    def __init__(self, tau, alpha, SNR, W, N, time_unit=1e01):
        """
        Inputs:
        -------
        tau : ndarray, size=(M, K)
            The delays between the BS and the UE for each of the M UE locations
            and each of the K multipaths.
        alpha : ndarray, size=(M, K)
            The channel coefficients between the BS and the UE for each of the M UE locations
            and each of the K multipaths.
        SNR : float
            The transmit signal-to-noise-ratio.
        W : float
            The bandwidth in Hz.
        N : int
            The number of subcarriers.
        time_unit : float, optional
            Scaling of the time unit. The default is 10 which means that the
            time unit is in deci-seconds.
        """
        super(Fisher_Information, self).__init__()

        ### Assign attributes from input ###
        self.tau = tau*time_unit
        self.alpha = alpha
        self.SNR = SNR
        self.W = W/time_unit
        self.N = N

        ### Other attribute assignments ###
        self.M, self.K = tau.shape # Number of UE locations and multipaths
        self.delta_f = self.W/self.N # Sub-carrier spacing

        max_delay_diff = 1/self.W # Max delay from LOS delay used in FIM analysis
        self.Kprime = self.K - np.sum((self.tau[:, :]-self.tau[:, 0, None]) > max_delay_diff, axis=1) # Nr. paths

        ### Assertions ###
        assert tau.shape[0] == alpha.shape[0],\
            "Input data dimensions are not consistent..."
        assert tau.shape[1] == alpha.shape[1],\
            "Input data dimensions are not consistent..."

    def delay_steering_vector(self, tau, n):
        """
        Compute steering vector given a delay tau (size=(K, K))
        and subcarrier indices n (size=(N,)).
        """
        dn_tau = np.exp(-1j*2*np.pi*(n[:, None, None]-1)*self.delta_f*tau[None, :, :])
        return dn_tau

    def Fisher_Information_Matrix(self):
        """
        Compute the Fisher information matrix USING simplified expressions.
        """
        FIM_list = list()
        n_pos = np.arange(1, int(self.N/2)+1)
        for TU_idx in range(self.M):
            Kidx = self.Kprime[TU_idx]
            tau_diff = self.tau[TU_idx, :Kidx, None] - self.tau[TU_idx, None, :Kidx]

            # A11
            const = (2*np.pi*self.delta_f)**2
            a_mult = np.real(self.alpha[TU_idx, :Kidx, None]*np.conjugate(self.alpha[TU_idx, None, :Kidx]))
            harmonic_sum = np.sum(n_pos[:, None, None]**2*np.real(self.delay_steering_vector(tau_diff, n_pos)), axis=0)
            A11 = 2*const*a_mult*harmonic_sum
            diag_const = self.N*(self.N+1)*(self.N+2)/12 * const
            #diag = np.array([np.abs(a)**2 for a in self.alpha[TU_idx, :]])*diag_const
            diag = np.abs(self.alpha[TU_idx, :])**2 * diag_const
            np.fill_diagonal(A11, diag) # in-place modification

            # A21
            harmonic_sum = np.sum(n_pos[:, None, None]*np.imag(self.delay_steering_vector(tau_diff, n_pos)), axis=0)
            A21 = -4*np.pi*self.delta_f*np.real(self.alpha[TU_idx, None, :Kidx])*harmonic_sum
            np.fill_diagonal(A21, np.zeros(Kidx))

            # A31
            # harmonic_sum = np.sum(n_pos[:, None, None]*np.imag(self.delay_steering_vector(tau_diff, n_pos)), axis=0)
            A31 = -4*np.pi*self.delta_f*np.imag(self.alpha[TU_idx, None, :Kidx])*harmonic_sum
            np.fill_diagonal(A31, np.zeros(Kidx))

            # A22
            A22 = 1 + 2*np.sum(np.real(self.delay_steering_vector(tau_diff, n_pos)), axis=0)
            np.fill_diagonal(A22, np.ones(Kidx)*(self.N+1)) # in-place modification

            # FIM
            combined_mat_col1 = np.concatenate((A11, A21, A31), axis=0)
            combined_mat_col2 = np.concatenate((A21.T, A22, np.zeros((Kidx, Kidx))), axis=0)
            combined_mat_col3 = np.concatenate((A31.T, np.zeros((Kidx, Kidx)), A22), axis=0)
            combined_mat = np.concatenate((combined_mat_col1, combined_mat_col2, combined_mat_col3), axis=1)
            FIM = 2*self.SNR * combined_mat
            # print(FIM)
            # print(2*10**(30/10) * combined_mat)
            # assert False
            FIM_list.append(FIM)
        return FIM_list

    def Equivalent_Fisher_information(self, FIM_list):
        """
        Compute the equivalent Fisher information for each of the Fisher
        information matrices in the list.
        """
        EFIM = np.zeros(self.M)
        for TU_idx, FIM in enumerate(FIM_list):
            # Cannot have negative Fisher information -> No information
            EFIM[TU_idx] = np.maximum(0, FIM[0, 0] - np.dot(FIM[0, 1:], np.linalg.solve(FIM[1:, 1:], FIM[1:, 0])))
        return EFIM

    def __call__(self):
        """
        Main class call. Runs the Fisher analysis for each of the UE locations.

        Returns
        -------
        EFIM : ndarray, size=(M,)
            The equivalent Fisher information.
        """
        FIM_list = self.Fisher_Information_Matrix()
        EFIM = self.Equivalent_Fisher_information(FIM_list)
        return EFIM

