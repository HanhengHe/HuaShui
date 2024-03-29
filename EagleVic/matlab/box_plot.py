import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图大小

df = pd.read_csv("allVFA.csv")
df.plot(kind='box', title='allVFA')
plt.show()


df = pd.read_csv("num1.csv")
df.plot(kind='box', title='1#VFA')
plt.show()


df = pd.read_csv("num2.csv")
df.plot(kind='box', title='2#VFA')
plt.show()


df = pd.read_csv("num3.csv")
df.plot(kind='box', title='3#VFA')
plt.show()

data = [0,2,3,4,5,6,7,10]
plt.boxplot(data, sym='o', whis=0.05)
plt.show()