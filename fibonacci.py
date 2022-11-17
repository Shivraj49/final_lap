
print("How many numbers you want to print ?")
tNum = int (input())
num1,num2,count= 0 ,1 ,0

if (tNum == 0):
    print("Valid Number")
    
elif (tNum == 1):
    print(num1)
    
else:
    while(count< tNum):
        print(num1, end=" ")
        c = num1 + num2
        num1 = num2
        num2 = c
        count += 1
        


