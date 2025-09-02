from tdda.rexpy import rexpy

x = rexpy.extract(['EH7 8BQ', 'W1 1AA', 'G5 1NN', 'AL7 3RT', 'EH7 8BQ'],
                  as_object=True, verbose=2)
print(x)
