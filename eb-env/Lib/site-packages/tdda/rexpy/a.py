from __future__ import print_function
from artists.giacometti.utils import Timer
t = Timer()
N = (10 * 1000 * 1000)
a = [None] * N
print(t.SplitTimeStr('[None] * N: '))
a = [1] * N
print(t.SplitTimeStr('[1] * N: '))
a = [None for i in range(N)]
print(t.SplitTimeStr('[None for i in range(N)]: '))
a = ['' for i in range(N)]
print(t.SplitTimeStr("['' for i in range(N)]: "))
a = [1 for i in range(N)]
print(t.SplitTimeStr("[1 for i in range(N)]: "))

