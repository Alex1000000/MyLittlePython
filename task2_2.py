def longestIncreasingSubsequence(X):
    N = len(X)
    Prev = [0] * N
    M = [0] * (N+1)
    Length = 0
    for i in range(N):
        if (not isinstance(X[i], int )):
            raise Exception('There is real number in the array.')
        lower = 1
        upper = Length
        while lower <= upper:
            mid = (lower + upper) // 2
            if (X[M[mid]] < X[i]):
                lower = mid + 1
            else:
                upper = mid - 1
        newL = lower
        Prev[i] = M[newL - 1]
        M[newL] = i

        if (newL > Length):
            Length = newL
    return Length


A=[0, 1,2,4,5,0,1,2,3]
print( longestIncreasingSubsequence(A))

