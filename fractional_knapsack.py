
def maxim (piwi):
    c=0
    maxi=piwi[0]
    for m in range (len(piwi)):
        if (piwi[m] > maxi):
            maxi = piwi[m]
            c=m
          
    if (piwi[c] > piwi[0]):
        return c
    
    else:
        return 0
    
    
print("Enter no. of items ")
n = int (input())
profit=[]
weight=[]
print("Enter Profit for each item")
for i in range (n):
    a = int(input())
    profit.append(a)
    
print("Enter Weight for each item")
for j in range (n):
    a=int(input())
    weight.append(a)
    
print("Enter knapsack capacity")
cap=int(input())

# n = 4
# profit=[280,100,120,120]
# weight=[40,10,20,24]
# cap =60

piwi=[]

for k in range (n):
    piwi.append(profit[k]/weight[k])
    
fProfit=0.0
for l in range (n):
    result = maxim(piwi)
    
    if (cap >= weight[result]):
        cap = cap - weight[result]
        
        fProfit= fProfit + profit[result]
        
        weight.pop(result)
        profit.pop(result)
        piwi.pop(result)
        
        
    else:
        fProfit += profit[result] * cap / weight[result]
        break
    
print(fProfit)
