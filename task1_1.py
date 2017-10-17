s = input("Enter numbers in range (1,N): ")
numbers = list(map(int, s.split()))
N=max(numbers)
count=0
mask=[0]*N
for i in numbers:
    mask[i - 1] += 1
for i in mask:
    if (i>0):
        count+=1
print(mask)
print("ANSWER= ",count)