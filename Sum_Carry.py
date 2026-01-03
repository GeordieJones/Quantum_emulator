from quantumComputer import QuantumC, GATES

gates = GATES
qc = QuantumC(4)

def sum_carry(input_index1, input_index2, sum_index, carry_index):
    #carry input if both controls are 1 then 3 will be 1
    qc.apply_gate(gates['TOFFOLI'], [input_index1, input_index2,carry_index])

    #this acts as the XOR
    qc.apply_gate(gates['CNOT'], [input_index1, sum_index]) #if row 1 is 1 then it flips 4
    qc.apply_gate(gates['CNOT'], [input_index2, sum_index]) # will flip row 4 if row 2 is 1
    
    sum_ = qc.measure_qubit(sum_index)
    carry = qc.measure_qubit(carry_index)

    return carry, sum_

def reset_inputs(qc, a, b):
    qc.state[:] = 0
    qc.state[(a << 0) | (b << 1)] = 1.0 + 0j


test_cases = {
    (0,0),
    (0,1),
    (1,0),
    (1,1)
}

# mapping
a_idx = 0
b_idx = 1
sum_idx = 2
carry_idx = 3

for a, b in test_cases:
    qc = QuantumC(4)
    reset_inputs(qc,a,b)
    carry, sum_ = sum_carry(a_idx, b_idx, sum_idx, carry_idx)
    print(f"Input a={a}, b={b} -> Sum={sum_}, Carry={carry}")
