from .OptimalFaultTolerance import OptimalFaultTolerance
import numpy as np
from itertools import combinations, product, compress, tee, chain

class MeasFaultTolerance(OptimalFaultTolerance):
    """
    MeasFaultTolerance class is a subclass of OptimalFaultTolerance class.
    This class is used to generate and evaluate possible error scenarios
    in a fault-tolerant quantum error correction scheme.
    """

    @staticmethod
    def _error_selector(errors):
        """
        Filters the errors based on whether all error events are unique or not.

        Parameters
        ----------
        errors : iterable
            An iterable collection of tuples where each tuple represents an error event.

        Yields
        ------
        bool
            True if all error events in a tuple are unique, False otherwise.
        """
        for error in errors:
            # Extract the list of control-X gate indices associated with each error event
            cx_list = [err[1] for err in error]
            # Yield True if all error events are unique, False otherwise
            if len(cx_list) == len(set(cx_list)):
                yield True
            else:
                yield False

    def _syndrome_generator_old(self, max_t, n_errors):
        """
        Generates syndromes for a given number of error events and time steps.

        Parameters
        ----------
        max_t : int
            Maximum number of time steps.
        n_errors : int
            Number of error events.

        Yields
        ------
        tuple
            The current time step, the array of syndrome values, and the array of error events.
        """
        # Define necessary variables
        n, k = self.n, self.k
        n_cx = self.n_cx
        n_total = self.n_total

        # Initialize arrays to store the syndrome and error events
        array = np.zeros(n_total-n, dtype=int)
        err_array = np.zeros(2*n, dtype=int)

        # Iterate over all time steps
        for dd in range(1, max_t+1):
            # Generate all possible error events
            meas_errors = ((0,-i) for i in range(1,1+self.n_meas))
            errors = combinations(chain(product(range(1, n_errors+1), range(n_cx)), meas_errors), dd)
            errors_1, errors_2 = tee(errors, 2)
            # Filter out error events with unique error positions
            selector = self._error_selector(errors_2)
            case = compress(errors_1, selector)

            # Process each possible error scenario
            for error_nums in case:
                for ind_err, ind_cx in error_nums:
                    # Update the syndrome and error arrays based on the error events
                    if ind_err == 0:
                        syndrome = np.zeros_like(array)
                        syndrome[-ind_cx-1] = 1
                        array = (array + syndrome) % 2
                    else:
                        syndrome = self.syndrome_power_set(ind_err, ind_cx)
                        error = self.error_power_set(ind_err, ind_cx)
                        array = (array + syndrome) % 2
                        err_array = (err_array + error) % 2
                # Yield the current time step and the updated syndrome and error arrays
                yield dd, array, err_array
                array[:] = 0
                err_array[:] = 0

    def syndrome_generator(self, max_t, n_errors, prep_errors=True):
        """
        Generates syndromes for a given number of error events and time steps.

        Parameters
        ----------
        max_t : int
            Maximum number of time steps.
        n_errors : int
            Number of error events.
        prep_errors : bool, optional
            If True, prep errors are included. Default is True.

        Yields
        ------
        tuple
            The current time step, the array of syndrome values, and the array of error events.
        """
        # Define necessary variables
        n, k = self.n, self.k
        n_cx = self.n_cx
        n_total = self.n_total
        c = 2 if prep_errors else 1

        # Initialize arrays to store the syndrome and error events
        array = np.zeros(n_total-n, dtype=int)
        err_array = np.zeros(2*n, dtype=int)

        # Iterate over all time steps
        for dd in range(1, max_t+1):
            # Generate all possible combinations of error events and operators
            combos = combinations(range(-c*self.n_meas, n_cx), dd)
            all_ops = product(range(1, n_errors+1), repeat=dd)
            case = product(all_ops, combos)

            # Process each possible error scenario
            for error_nums, inds in case:
                for ind_err, ind_cx in zip(error_nums, inds):
                    if ind_cx < 0:
                        # Handle prep_errors scenario
                        if prep_errors and ind_cx < -self.n_meas:
                            ind_cx += self.n_meas
                        syndrome = np.zeros_like(array)
                        syndrome[-ind_cx-1] = 1
                        array = (array + syndrome) % 2
                    else:
                        # Handle normal errors scenario
                        syndrome = self.syndrome_power_set(ind_err, ind_cx)
                        array = (array + syndrome) % 2
                        error = self.error_power_set(ind_err, ind_cx)
                        err_array = (err_array + error) % 2
                # Yield the current time step and the updated syndrome and error arrays
                yield dd, array, err_array
                array[:] = 0
                err_array[:] = 0