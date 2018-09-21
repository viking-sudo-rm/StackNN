"""
A script to demonstrate testing mode.
"""
from tasks import ReverseTask

task = ReverseTask(load_path="saved_models/linear_reverse_task_20.dat")
task.run_test("data/testing/reverse_length12.csv",
              log_file="log/test_results/reverse_length12_test.csv")
