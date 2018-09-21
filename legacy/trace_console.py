"""
A script to demonstrate the proper usage of Task.trace_console.
"""
from tasks import ReverseTask

task = ReverseTask()
task.run_experiment()
task.trace_console()
