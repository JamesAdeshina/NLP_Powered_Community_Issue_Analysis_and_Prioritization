import streamlit as st
import time
from concurrent.futures import ThreadPoolExecutor


def task_1():
    time.sleep(3)
    return "Task 1 complete"


def task_2():
    time.sleep(2)
    return "Task 2 complete"


def task_3():
    time.sleep(1)
    return "Task 3 complete"


# Function to run tasks in parallel
def run_tasks():
    with ThreadPoolExecutor() as executor:
        future1 = executor.submit(task_1)
        future2 = executor.submit(task_2)
        future3 = executor.submit(task_3)

        # Use futures to get results as they complete
        for future in [future1, future2, future3]:
            result = future.result()  # Wait for each task to finish
            st.write(result)


# Start tasks in parallel and display results incrementally
run_tasks()
