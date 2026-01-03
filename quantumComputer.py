import torch
import math

device = torch.device('cpu')

GATES = {
    # Single-qubit gates
    'I': torch.tensor([[1,0],[0,1]], dtype=torch.complex64),
    'X': torch.tensor([[0,1],[1,0]], dtype=torch.complex64),
    'Y': torch.tensor([[0,-1j],[1j,0]], dtype=torch.complex64),
    'Z': torch.tensor([[1,0],[0,-1]], dtype=torch.complex64),
    'H': (1 / math.sqrt(2)) * torch.tensor([[1,1],[1,-1]], dtype=torch.complex64),
    'S': torch.tensor([[1,0],[0,1j]], dtype=torch.complex64),
    'T': torch.tensor([[1,0],[0,torch.exp(torch.tensor(1j*math.pi/4, dtype=torch.complex64))]], dtype=torch.complex64),

    # Two-qubit gates
    'CNOT': torch.tensor([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,0,1],
        [0,0,1,0]
    ], dtype=torch.complex64),

    'CZ': torch.tensor([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,-1]
    ], dtype=torch.complex64),

    'SWAP': torch.tensor([
        [1,0,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,0,1]
    ], dtype=torch.complex64),

    # Three-qubit gate
    'TOFFOLI': torch.tensor([
        [1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,1,0]
    ], dtype=torch.complex64)
}

class QuantumC:
    def __init__(self, n, device='cpu'):
        self.n = n
        self.device = torch.device(device)
        # Initialize |0...0> state
        self.state = torch.zeros(2**n, dtype=torch.complex64, device=self.device)
        self.state[0] = 1.0 + 0j

    def apply_gate(self, gate, qubits):
        # gate -> gate_type; qubits -> [control, target/control2, target]
        k = len(qubits)
        state_tensor = self.state.view([2]*self.n)
        gate = gate.view([2]*2*k)
        dims = (list(range(k, 2*k)), qubits)
        new_state = torch.tensordot(gate, state_tensor, dims=dims)
        new_state = torch.movedim(new_state,tuple(range(k)), tuple(qubits))
        self.state = new_state.reshape(2**self.n)



    def measure_qubit(self, qubit):
        n = self.n
        state_t = self.state.view([2]*n)

        # probabilities where qubit=0 and qubit=1
        slicer0 = [slice(None)]*n
        slicer0[qubit] = 0
        prob0 = torch.sum(torch.abs(state_t[tuple(slicer0)])**2).item()

        # sample
        r = torch.rand(1).item()
        outcome = 0 if r < prob0 else 1

        # collapse
        slicer = [slice(None)]*n
        slicer[qubit] = outcome
        new_state = torch.zeros_like(state_t)
        new_state[tuple(slicer)] = state_t[tuple(slicer)]
        # renormalize
        norm = torch.sqrt(torch.sum(torch.abs(new_state)**2))
        if norm > 0:
            new_state = new_state / norm
        self.state = new_state.reshape(2**n)
        return int(outcome)


    def get_state(self):
        return self.state
    


# New Class testing


# 1-qubit circuit
qc1 = QuantumC(1)
qc1.apply_gate(GATES['H'], [0])
print("1-qubit H gate state: \nExpected: [0.7071+0.j, 0.7071+0.j]\nGot:", qc1.get_state())
# Expected: [0.7071+0.j, 0.7071+0.j]


# 2-qubit circuit, initialize |10> = a=1, b=0
qc2 = QuantumC(2)
qc2.state[0b10] = 1.0
qc2.state[0b00] = 0.0
qc2.apply_gate(GATES['CNOT'], [0,1])
print("2-qubit CNOT state:\nExpected: |11> → index 3\nGot:", qc2.get_state())
# Expected: |11> → index 3


# 3-qubit circuit, initialize |110> = a=1, b=1, carry=0
qc3 = QuantumC(3)
qc3.state[0b110] = 1.0
qc3.state[0b000] = 0.0
qc3.apply_gate(GATES['TOFFOLI'], [0,1,2])
print("3-qubit Toffoli state:\nExpected: |111> → index 7\nGot:", qc3.get_state())
# Expected: |111> → index 7