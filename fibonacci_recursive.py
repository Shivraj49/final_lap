def fib (n):
    
    if (n <= 1):
        return n
    
    else:
        return fib(n-1)+fib(n-2)
        

print("How many numbers you want to print ?")
n = int (input())

for i in range (n):
    print(fib(i), end= " ")