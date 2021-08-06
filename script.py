import multiprocessing
import os

# Creating the tuple of all the processes
all_processes = ('baseline.4.py', 'baseline.8.py',
                 'baseline.20.py', 'baseline.50.py')


# This block of code enables us to call the script from command line.
def execute(process):
    os.system(f'python {process}')


process_pool = multiprocessing.Pool(processes=4)
process_pool.map(execute, all_processes)
