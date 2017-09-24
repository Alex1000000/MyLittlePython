s = input("Enter numbers in range (1,N): ")
numbers = list(map(int, s.split()))
N=max(numbers)
print(N)
count=0
mask=[0]*N
for i in numbers:
    mask[numbers[count] - 1] += 1
    count += 1
count=0
for i in mask:
    if (i>0):
        count+=1
print("ANSWER= ",count)