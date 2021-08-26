import multiprocessing
import os

# Creating the tuple of all the processes
# all_processes = ('bl.40000.05.py', 'bl.40000.055.py',
#                  'bl.40000.06.py', 'bl.40000.065.py',
#                  'bl.40000.07.py', 'bl.40000.075.py',
#                  'bl.40000.08.py', 'bl.40000.085.py',
#                  'bl.40000.09.py', 'bl.40000.095.py')

all_processes = ('80000.2000.04.py', '80000.3000.04.py', '80000.4000.04.py', '80000.5000.04.py', '80000.7000.04.py',
                 '80000.3000.05.py', '80000.7000.05.py', '80000.10000.05.py', '80000.14000.05.py',
                 '80000.14000.06.py', '80000.16000.06.py', '80000.18000.06.py',
                 '80000.18000.07.py', '80000.20000.07.py')

# This block of code enables us to call the script from command line.


def execute(process):
    os.system(f'python {process}')


process_pool = multiprocessing.Pool(processes=8)
process_pool.map(execute, all_processes)
