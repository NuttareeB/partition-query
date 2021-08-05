import multiprocessing
import os

# Creating the tuple of all the processes
all_processes = ('100.4.20.py', '100.4.20.py', '100.4.20.py', '100.4.20.py')


# This block of code enables us to call the script from command line.
def execute(process):
    os.system(f'python {process}')


process_pool = multiprocessing.Pool(processes=4)
process_pool.map(execute, all_processes)