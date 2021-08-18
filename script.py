import multiprocessing
import os

# Creating the tuple of all the processes
# all_processes = ('bl.40000.05.py', 'bl.40000.055.py',
#                  'bl.40000.06.py', 'bl.40000.065.py',
#                  'bl.40000.07.py', 'bl.40000.075.py',
#                  'bl.40000.08.py', 'bl.40000.085.py',
#                  'bl.40000.09.py', 'bl.40000.095.py')

all_processes = ('40000.500.05.py', '40000.500.055.py',
                 '40000.1000.05.py', '40000.1000.055.py',
                 '40000.1000.06.py', '40000.1000.065.py',
                 '40000.1500.06.py', '40000.1500.065.py',
                 '40000.1500.07.py', '40000.1500.075.py',
                 '40000.1500.08.py', '40000.1500.085.py',
                 '40000.2000.07.py', '40000.2000.075.py',
                 '40000.2000.08.py', '40000.2000.085.py',
                 '40000.2000.09.py', '40000.2000.095.py',
                 '40000.2500.09.py', '40000.2500.095.py',)

# This block of code enables us to call the script from command line.


def execute(process):
    os.system(f'python {process}')


process_pool = multiprocessing.Pool(processes=20)
process_pool.map(execute, all_processes)
