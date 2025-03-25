import streamlit as st
import asyncio
import time


async def task_1():
    await asyncio.sleep(3)
    return "Task 1 complete"


async def task_2():
    await asyncio.sleep(2)
    return "Task 2 complete"


async def task_3():
    await asyncio.sleep(1)
    return "Task 3 complete"


# Function to run all tasks asynchronously
async def run_tasks():
    task1 = asyncio.create_task(task_1())
    task2 = asyncio.create_task(task_2())
    task3 = asyncio.create_task(task_3())

    result_1 = await task1
    st.write(result_1)

    result_2 = await task2
    st.write(result_2)

    result_3 = await task3
    st.write(result_3)


# Run tasks asynchronously
asyncio.run(run_tasks())
