from .DegenerateFaultTolerance import DegenerateFaultTolerance
from .surface import random_surface_code
from qiskit import QuantumRegister, QuantumCircuit

class SurfaceFaultTolerance(DegenerateFaultTolerance):

    def __init__(self,
                L,
                gate_error_list = None,
                apply_stabilizer_errors = True):

        self.L = L
        self.n, self.k = L**2 + (L-1)**2, 1
        self.qr = QuantumRegister(self.n, 'q')
        self.anc = QuantumRegister(self.n-self.k, 'a')
        self._set_gate_error_list(gate_error_list)
        self.full_circuit = None
        self.apply_encoding_errors = False # It can't be used here.
        self.apply_stabilizer_errors = apply_stabilizer_errors
        self._set_bin_rep()

    def get_encoding(self):
        self.stabilizers = random_surface_code(self.L)
        #print(self.stabilizers)
        self.encoding = QuantumCircuit(self.qr)
