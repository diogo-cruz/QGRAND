import numpy as np
from .pauli import pauli_generator
from QGRAND import QGRAND
import sys
import matplotlib.pyplot as plt
plt.rc('text',usetex=True)
plt.rc('font', family='serif', size=24)
plt.rcParams['figure.figsize'] = [10, 6]

import numpy as np
import matplotlib.pyplot as plt

class DecisionTree:
    """
    A decision tree for quantum error correction.
    """

    def __init__(self, x, logN, n, max_t):
        """
        Initialize the decision tree with parameters.

        Parameters
        ----------
        x : int
            Parameter related to number of iterations.
        logN : float
            Parameter related to the number of errors.
        n : int
            Number of iterations.
        max_t : int
            Maximum error correction depth.
        """
        self.x = x
        self.logN = logN
        self.n = n
        self.max_t = max_t
        self.errors = []
        self._set_default_error()
        self._set_min_iterations()

    def _set_default_error(self):
        """
        Set default error values for the decision tree.
        """
        # Generate the default errors
        errors = [''.join(['I' for _ in range(self.n)])]
        errors += [error for _, error in pauli_generator(self.n, self.max_t+1)]
        self.errors = errors

    def _set_min_iterations(self):
        """
        Set minimum iterations for error correction in the decision tree.
        """
        # Calculate the minimum iterations required for error correction
        n_errors = len(self.errors)
        min_iters = self.lower_bound_iters(n_errors, self.n-self.k)
        min_iters_sim, q_p, q_m = self.lower_bound_sim(n_errors, self.n-self.k, runs=10_001)

    def get_number_of_iterations(self):
        """
        Calculate the number of iterations for error correction.
        """
        # Calculate the number of iterations required for each logN value
        self.n_iters = [[self.lower_bound_iters(float(10**j), i) for i in 2**(2**self.x)] for j in self.logN]

    def plot_number_of_iterations(self, save = False, filename = 'decision_tree_savings'):
        """
        Plot the number of iterations against logN.

        Parameters
        ----------
        save : bool, optional
            Whether to save the plot.
        filename : str, optional
            The filename of the plot if saved.
        """
        fig = plt.figure()
        for i, plot in enumerate(self.n_iters):
            plt.plot(self.x, plot, label='$N=10^{}$'.format(self.logN[i]))
        plt.legend()
        plt.ylabel('Number of iterations')
        plt.xlabel('$\log_2 \log_2 s$')
        plt.show()

        # Save the plot if required
        if save:
            filename = 'Plots/'+filename+'.pdf'
            fig.savefig(filename, bbox_inches='tight')

    def get_fit(self):
        """
        Calculate the fit for the number of iterations.

        Returns
        -------
        soln : array
            Polynomial coefficients.
        """
        data = np.array(self.n_iters).T
        N_array = np.log2(10**self.logN)
        return self.polyfit2d(N_array, self.x, data, kx=1, ky=1, order=1)

    @staticmethod
    def lower_bound_iters(N, S):
        """
        Calculate the lower bound of iterations.
        
        Parameters
        ----------
        N : int
            The initial count.
        S : int
            The decrement count.
            
        Returns
        -------
        n_iters : int
            The lower bound of iterations.
        """
        N -= 1
        f = lambda N, S: N/2 - np.sqrt(N * np.log(S) / 2)  # Function to calculate new N
        n_iters = 0
        while N > 0.5:
            N = f(N, S-n_iters)
            n_iters += 1
        return n_iters


    @staticmethod
    def lower_bound_sim(N, S, runs=31, conf=0.2):
        """
        Simulate to get the lower bound with a random set of data.

        Parameters
        ----------
        N : int
            Number of rows in the data.
        S : int
            Number of columns in the data.
        runs : int, optional
            Number of simulations to run.
        conf : float, optional
            Confidence level for quantile calculation.

        Returns
        -------
        median : float
            Median of iterations from the simulations.
        lower_quantile : float
            Lower quantile of iterations from the simulations.
        upper_quantile : float
            Upper quantile of iterations from the simulations.
        """
        N -= 1
        iter_array = np.zeros(runs, dtype=int)
        for run in range(runs):
            min_data_sum = 0
            while min_data_sum == 0:  # Ensure at least one positive sum in data
                data = np.random.randint(2, size=(N, S))
                min_data_sum = np.min(np.sum(data, axis=1))
            iters = 0
            while data.shape[0] > 0:  # Keep reducing data until no rows are left
                next_stabilizer = np.argmax(np.sum(data, axis=0))  # Identify the next stabilizer
                good_rows = data[:, next_stabilizer] == 0  # Identify good rows where next_stabilizer is 0
                data = data[good_rows]  # Keep only good rows in data
                iters += 1
            iter_array[run] = iters
        return np.median(iter_array), np.quantile(iter_array, 1-conf), np.quantile(iter_array, conf)


    @staticmethod
    def shannon_entropy(p, probs):
        """
        Calculate the Shannon entropy of a given probability distribution.

        Parameters
        ----------
        p : array
            The array of probabilities.
        probs : array
            The array of event probabilities.

        Returns
        -------
        entropy : float
            The Shannon entropy.
        prob_1 : array
            The sum of product of p and probs.
        """
        probs /= np.sum(probs)  # Normalize probabilities
        prob_1 = np.sum(p * probs[:,None], axis=0)
        entropy = -np.nan_to_num(prob_1*np.log2(prob_1)) - np.nan_to_num((1-prob_1)*np.log2(1-prob_1))  # Calculate entropy
        return entropy, prob_1

    def get_decision_tree(self, measured_stabilizers=None, measurements=None, prob=1.):
        """
        Returns a decision tree for the measurement of stabilizers.

        Parameters
        ----------
        measured_stabilizers : np.array, optional
            The stabilizers that have been measured, by default None.
        measurements : np.array, optional
            The measurements that have been taken, by default None.
        prob : float, optional
            The probability of the current branch, by default 1..

        Returns
        -------
        list
            The decision tree for the measurement of stabilizers.
        """
        # Initialize the measured stabilizers and measurements if not provided
        if measured_stabilizers is None:
            measured_stabilizers = np.array([], dtype=int)
            measurements = np.array([], dtype=int)
            good_rows = np.full(syndrome_table.shape[0], True, dtype=bool)
        else:
            # Find the rows in the syndrome table that match the measurements
            good_rows = np.squeeze(np.all(syndrome_table[:, measured_stabilizers] == measurements, axis=1))

        # Reduce the syndrome table to only rows that match the measurements
        working_table = syndrome_table[good_rows]
        assert len(measured_stabilizers) <= syndrome_table.shape[1], 'Stuck in loop.'

        if working_table.shape[0] <= 1:
            index = np.argwhere(good_rows)[0,0] if working_table.shape[0] == 1 else -1
            return [(index, len(measured_stabilizers))]

        # Calculate the entropy and probabilities of the working table
        entropy, prob_1 = self.shannon_entropy(working_table, noise_probabilities[good_rows])

        # Mask already measured stabilizers
        mask = np.zeros(entropy.size, dtype=bool)
        mask[measured_stabilizers] = True
        masked_entropy = np.ma.array(entropy, mask=mask)

        # Find the next stabilizer to measure
        next_stabilizer = np.argmax(masked_entropy)
        assert next_stabilizer not in measured_stabilizers, "Repeated stabilizer."
        measured_stabilizers = np.append(measured_stabilizers,[next_stabilizer])

        # Update probability
        prob_1 = prob_1[next_stabilizer]

        # Recurse on the 0 and 1 branches of the decision tree
        data_0 = self.get_decision_tree(measured_stabilizers, np.append(measurements,[0]), prob*(1-prob_1))
        data_1 = self.get_decision_tree(measured_stabilizers, np.append(measurements,[1]), prob*prob_1)

        return data_0 + data_1

    def shannon(alpha, N, eps=None):
        """
        Returns the Shannon entropy and probabilities for a given alpha and N.

        Parameters
        ----------
        alpha : float
            The scaling parameter for the exponential distribution.
        N : int
            The number of terms to calculate in the exponential distribution.
        eps : float, optional
            The perturbation added to the exponential distribution, by default None.

        Returns
        -------
        tuple
            The Shannon entropy and probabilities.
        """
        # Calculate the probabilities based on whether eps is provided or not
        if eps is None:
            probs = np.exp(-alpha*np.arange(N))
            probs /= np.sum(probs)
        else:
            probs = np.full(N, eps/(N-1))
            probs[0] = 1 - eps 
            
        # Calculate the entropy of the distribution
        entropy = -np.sum(probs*np.log2(probs))   

        return entropy, probs

    def entropy_dependence(self, n, k, ty, alpha, eps, runs=3):
        """
        Calculate entropy dependence on errors in quantum error correction.

        Parameters
        ----------
        n : int
            The number of quantum bits.
        k : int
            The number of logical qubits.
        ty : int
            Type of error.
        alpha : float
            Parameter for Shannon entropy.
        eps : float
            Epsilon for constant probabilities in Shannon entropy.
        runs : int, optional
            Number of runs for the simulation, by default 3.

        Returns
        -------
        tuple
            A tuple containing entropy, epsilon entropy, max iterations, min iterations, min iterations simulation,
            q_p, q_m, iterations array, epsilon iterations array.
        """
        
        # Setting up global variables
        global syndrome_table
        global noise_probabilities
        
        # Increase ty by 1 as indices start from 0
        ty += 1
        
        # Initialize errors with a string of 'I's
        errors = [''.join(['I' for _ in range(n)])]
        
        # Add errors from pauli_generator
        errors += [error for _, error in pauli_generator(n, ty)]
        
        # Get the number of errors
        n_errors = len(errors)
        
        # Lower bound iterations based on the number of errors and logical qubits
        min_iters = self.lower_bound_iters(n_errors, n-k)
        
        # Simulate lower bound iterations
        min_iters_sim, q_p, q_m = self.lower_bound_sim(n_errors, n-k, runs=10_001)
        
        # Calculate Shannon entropy and maximum iterations
        _, probs = self.shannon(alpha, n_errors)
        entropy, max_iters = -np.sum(np.nan_to_num(probs*np.log2(probs))), np.log2(probs.shape[0])
        
        # Create noise statistics
        noise = [(prob, error) for prob, error in zip(probs, errors)]
        
        # Initialize array for storing iterations data
        iter_array = np.zeros(runs)
        
        # Simulation loop
        for run in range(runs):
            # Create QGRAND trial with noise
            trial = QGRAND(n=n, k=k, num_gates=5000, noise_statistics=noise)
            
            # Get encoding and syndrome table
            trial.get_encoding()
            trial.get_syndrome_table()
            syndrome_table = trial.syndrome_table
            noise_probabilities = np.array(trial.noise_probabilities)
            
            # Ensure uniqueness of syndrome table
            syndrome_table, index = np.unique(syndrome_table, axis=0, return_index=True)
            noise_probabilities = noise_probabilities[index]

            # Get decision tree iterations data and store it in the array
            iter_data = self.get_decision_tree()
            iter_array[run] = sum([iters * noise_probabilities[i] for i, iters in iter_data])
            
        # Calculate entropy with constant probabilities (epsilon)
        _, probs = self.shannon(alpha, n_errors, eps=eps)
        entropy_eps = -np.sum(np.nan_to_num(probs*np.log2(probs)))
        
        # Create noise statistics with constant probabilities
        noise = [(prob, error) for prob, error in zip(probs, errors)]
        
        # Initialize array for storing iterations data with epsilon
        iter_array_eps = np.zeros(runs)
        
        # Simulation loop with epsilon
        for run in range(runs):
            # Create QGRAND trial with noise
            trial = QGRAND(n=n, k=k, num_gates=5000, noise_statistics=noise)

            # Get encoding and syndrome table
            trial.get_encoding()
            trial.get_syndrome_table()
            syndrome_table = trial.syndrome_table
            noise_probabilities = np.array(trial.noise_probabilities)

            # Ensure uniqueness of syndrome table
            syndrome_table, index = np.unique(syndrome_table, axis=0, return_index=True)
            noise_probabilities = noise_probabilities[index]

            # Get decision tree iterations data and store it in the array
            iter_data = self.get_decision_tree()
            iter_array_eps[run] = sum([iters * noise_probabilities[i] for i, iters in iter_data])
        
        # Return calculated entropy values, iterations and other related data
        return entropy, entropy_eps, max_iters, min_iters, min_iters_sim, q_p, q_m, iter_array, iter_array_eps

    @staticmethod
    def polyfit2d(x, y, z, kx=3, ky=3, order=None):
        """
        Perform two dimensional polynomial fitting by least squares.

        This function fits the functional form f(x,y) = z.

        Parameters
        ----------
        x : array-like, 1d
            x coordinates.
        y : array-like, 1d
            y coordinates.
        z : np.ndarray, 2d
            Surface to fit.
        kx : int, optional
            Polynomial order in x. Default is 3.
        ky : int, optional
            Polynomial order in y. Default is 3.
        order : int or None, optional
            If None, all coefficients up to maximum kx, ky, ie. up to and 
            including x^kx*y^ky, are considered. If int, coefficients up to a 
            maximum of kx+ky <= order are considered. Default is None.

        Returns
        -------
        soln : np.ndarray
            Array of polynomial coefficients.
        residuals : np.ndarray
        rank : int
        s : np.ndarray

        Notes
        -----
        The resultant fit can be plotted with:
        np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))
        """

        # Generate grid of coordinates for x and y
        x, y = np.meshgrid(x, y)
        
        # Initialize the coefficient array with ones, up to x^kx, y^ky
        coeffs = np.ones((kx+1, ky+1))

        # Initialize the array to solve with zeros
        a = np.zeros((coeffs.size, x.size))

        # For each coefficient, produce array x^i, y^j
        for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
            
            # Do not include powers greater than order if order is set
            if order is not None and i + j > order:
                arr = np.zeros_like(x)
            else:
                arr = coeffs[i, j] * x**i * y**j
            
            a[index] = arr.ravel()

        # Solve using least squares fitting and return the result
        return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)
