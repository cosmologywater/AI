
import os

nowstr = os.popen('echo $OMP_NUM_THREADS').read()
print(nowstr)
