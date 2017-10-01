def power(a,n):
    if n==1:
        return a
    elif n%2==0:
        return power(a,n/2)*power(a,n/2)
    else:
        return a*power(a,n-1)

print(power(2,3))