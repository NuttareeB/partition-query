import multiprocessing
import os

# Creating the tuple of all the processes
all_processes = ('1000.25.20.py', '1000.25.50.py',
                 '1000.25.100.py', '1000.25.150.py',
                 '1000.25.200.py', '1000.25.300.py')


# This block of code enables us to call the script from command line.
def execute(process):
    os.system(f'python {process}')


process_pool = multiprocessing.Pool(processes=6)
process_pool.map(execute, all_processes)
