# Linear-Block-Code
# Aim
Write a simple python program to Generate Matrix, Codeword, Hamming weight, Syndrome matrix and find the error on received codeword using Linear block code. 
# Tools required
IDE python with scipy and numpy
# Program
```
import numpy as np

pb = []      # Parity matrix P
h_dis = []   # Hamming distances
r_code = []  # Received code
err = []     # Error vector

# Input
col = int(input("Enter the Parity bits (n-k): "))
row = int(input("Enter the Message bits (k): "))

# Input Parity matrix (P)
for i in range(row):
    p = list(map(int, input(f"Enter the row values {i+1} (Separated by space): ").split()))
    if len(p) != col:
        print("Error: Number of columns doesn't match the entered parity bit count.")
        exit()
    pb.append(p)

p_mat = np.array(pb, dtype=int)
Ik = np.eye(row, dtype=int)
g_mat = np.hstack((p_mat, Ik))  # G = [P | I]

# Dimensions
n, k = g_mat.shape[1], g_mat.shape[0]

# Generate all possible messages (2^k)
m = np.array([[1 if (i >> (k - j - 1)) & 1 else 0 for j in range(k)] for i in range(2 ** k)])
# Generate codewords: c = m * G mod 2
c = np.mod(np.dot(m, g_mat), 2)

# Hamming weight calculation
for row in c:
    h_dis.append(np.sum(row))
h_mat = np.array(h_dis).reshape(-1, 1)

# Minimum Hamming distance (excluding zero codeword)
d_min = np.min(np.sum(c[1:], axis=1))

# Parity check matrix H = [I | P^T]
h = p_mat[:, :col]
hp = np.hstack((np.eye(col, dtype=int), h.T))
ht = hp.T  # H transpose

# Add a zero row at top for syndrome table display
zero_row = np.zeros((1, ht.shape[1]), dtype=int)
hp1 = np.vstack((zero_row, ht))

# Output Generator matrix
print('\nThe Generator Matrix G = [P | I]:')
for r in g_mat:
    print(" ".join(map(str, r)))

# Display all messages, codewords and weights
print('\nMessage Bits  Codeword           Hamming Weight')
code_word = np.hstack((m, c, h_mat))
for r in range(code_word.shape[0]):
    message = " ".join(map(str, code_word[r, :k]))
    code = " ".join(map(str, code_word[r, k:k+n]))
    weight = str(code_word[r, -1])
    print(f'{message}\t{code}\t{weight}')

print(f'\nMinimum Hamming Distance (d_min): {d_min}')

# Display Parity-Check Matrix
print('\nParity Check Matrix H = [I | P^T]:')
for r in hp:
    print(" ".join(map(str, r)))

print('\nParity Check Matrix Transpose (H^T):')
for r in hp1:
    print(" ".join(map(str, r)))

# Input received codeword
rc = list(map(int, input(f"\nEnter the received codeword (space separated {n} bits): ").split()))
if len(rc) != n:
    print("Error: Length of received codeword must match the codeword length.")
    exit()

r_code.append(rc)
r_c = np.array(r_code)

# Syndrome calculation: s = r * H^T mod 2
e = np.mod(np.dot(r_c, ht), 2)
print(f"\nSyndrome of received codeword: {' '.join(map(str, e[0]))}")

# Find error vector by comparing syndrome with each column of H^T
for i in range(n):
    if np.array_equal(e[0], ht[i, :]):
        err = np.eye(n, dtype=int)[i, :]
        break
else:
    err = np.zeros(n, dtype=int)  # No error detected

print(f"Error position vector: {' '.join(map(str, err))}")

# Syndrome Table
print('\nSyndrome Table:')
n1 = hp1.shape[0]
for i in range(n1):
    combined_row = np.concatenate((hp1[i, :], np.eye(n1, dtype=int)[i, :]))
    formatted_row = " ".join(map(str, combined_row[:hp1.shape[1]])) + '\t' + " ".join(map(str, combined_row[hp1.shape[1]:]))
    print(formatted_row)

# Correct the received codeword
corrected = (np.array(rc) + err) % 2
print(f"\nCorrected codeword: {' '.join(map(str, corrected))}")
```
# Output Waveform
![image](https://github.com/user-attachments/assets/e26540e9-fcab-4288-b43a-8174cfcbeda9)
![image](https://github.com/user-attachments/assets/ce752e06-4c43-459e-9354-f3656a38e166)
# Results
```
Thus,The linear block code was successfully implemented for error detection and single-bit error correction.
```
# Hardware experiment output waveform.
