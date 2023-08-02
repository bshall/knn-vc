import numpy as np

def generate_matrix_from_index(A, len=25):
    matrix = np.zeros(len, dtype=float)
    matrix[A] = 1
    return matrix


def retrieve_index_from_matrix(matrix):
    A = np.where(matrix == 1)[0]
    return A

if __name__ == '__main__':
    # Generating a matrix from index A
    A = 6
    matrix = generate_matrix_from_index(A)
    print("Generated Matrix:")
    print(matrix)

    # Retrieving index A from the matrix
    retrieved_A = retrieve_index_from_matrix(matrix)
    print("Retrieved Index A:")
    print(retrieved_A)
