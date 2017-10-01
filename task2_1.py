def LCS(A,B):
    n_a=len(A)
    n_b=len(B)
    matrixOfLength=[[0]*(n_a+1) for i in range(n_b+1)]
    for i in range(1,n_b+1):
        for j in range(1,n_a+1):
            if A[j-1]==B[i-1]:
                matrixOfLength[i][j]=matrixOfLength[i-1][j-1]+1
            else:
                matrixOfLength[i][j]=max(matrixOfLength[i-1][j],matrixOfLength[i][j-1])
    # print(matrixOfLength)
    # print(matrixOfLength[n_b][n_a])
    return matrixOfLength[n_b][n_a]
A=[1,2,6,3,5]
B=[1,4,2, 5]
LCS(A,B)