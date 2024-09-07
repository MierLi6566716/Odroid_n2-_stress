import os
import time
import random
import multiprocessing as mp
from statistics import mean
import csv
import torch
import torch.nn as nn
import torch.nn.functional as f
import timm


def get_temps():
    """Reads temps of cores 4-7 on the A15 and returns them as an array."""
    temp = open(r"/sys/class/thermal/thermal_zone0/temp")
    temps = [int(temp.readline())/1000]
    temp.close()
    return temps

def log_temperature_data(mode, stop_event, interval=0.2):
    """Log the temperature data to a CSV file in real time."""
    start_time = time.time()
    file_path = f"/home/vr_stress/stress_{mode}.csv"
    print(f"Starting to log temperature data to {file_path}")

    with open(file_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time", "temp"])

        while not stop_event.is_set():
            temp = get_temps()
            current_time = time.time() - start_time
            writer.writerow([current_time, temp[0]])
            f.flush()  # Ensure the data is written to the file immediately
            print(f"Logged: time={current_time}, temp={temp[0]}")
            time.sleep(interval)
    
    print(f"Finished logging temperature data to {file_path}")


def matrix_multiply(dim):
    with torch.no_grad():
        input = torch.randn(8, 3, 224, 224)
        model = timm.create_model("resnet50")
        output = model(input)

    del input, output, model
    torch.cuda.empty_cache()

def cleanup():
    gc.collect()
    torch.cuda.empty_cache() 
    
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated()}")
    print(f"CUDA memory cached: {torch.cuda.memory_reserved()}")


def scheduler(mode="c"):
    """Runs processes concurrently (c) or sequentially (s)."""
    os.system(f"taskset -p -c 3 {os.getpid()}")

    p1 = mp.Process(target=matrix_multiply, args=(400,))
    p2 = mp.Process(target=matrix_multiply, args=(400,))
    p3 = mp.Process(target=matrix_multiply, args=(400,))
    p4 = mp.Process(target=matrix_multiply, args=(400,))

    log = []
    stop_event = mp.Event()

    if mode == "c":
        t0 = time.time()
        p1.start()
        p2.start()
        p3.start()
        p4.start()

        os.system(f"taskset -p -c 2 {p1.pid}")
        os.system(f"taskset -p -c 3 {p2.pid}")
        os.system(f"taskset -p -c 4 {p3.pid}")
        os.system(f"taskset -p -c 5 {p4.pid}")

        temp_logger = mp.Process(target=log_temperature_data, args=(mode, stop_event, 0.5))
        temp_logger.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()

        stop_event.set()
        temp_logger.join()

    elif mode == "s":
        t0 = time.time()

        temp_logger = mp.Process(target=log_temperature_data, args=(mode, stop_event, 0.5))
        temp_logger.start()

        p1.start()
        os.system(f"taskset -p -c 3 {p1.pid}")
        p1.join()

        p2.start()
        os.system(f"taskset -p -c 3 {p2.pid}")
        p2.join()

        p3.start()
        os.system(f"taskset -p -c 3 {p3.pid}")
        p3.join()

        p4.start()
        os.system(f"taskset -p -c 3 {p4.pid}")
        p4.join()

        stop_event.set()
        temp_logger.join()


def hybrid(t_min, t_max, mode="single", dim=512):
    os.system(f"taskset -p -c 3 {os.getpid()}")

    p1 = mp.Process(target=matrix_multiply, args=(dim,))
    p2 = mp.Process(target=matrix_multiply, args=(dim,))
    p3 = mp.Process(target=matrix_multiply, args=(dim,))
    p4 = mp.Process(target=matrix_multiply, args=(dim,))

    stop_event = mp.Event()
    file_path = f"/home/vr_stress/stress_{mode}.csv"
    
    t0 = time.time()
    temp_logger = mp.Process(target=log_temperature_data, args=(mode, stop_event, 0.5))
    temp_logger.start()

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    os.system(f"taskset -p -c 2 {p1.pid}")
    os.system(f"taskset -p -c 3 {p2.pid}")
    os.system(f"taskset -p -c 4 {p3.pid}")
    os.system(f"taskset -p -c 5 {p4.pid}")

    time.sleep(5)

    while p1.is_alive() or p2.is_alive() or p3.is_alive() or p4.is_alive():
        temp = get_temps()[0]
        print("Enter loop")
        print(f"This is the current temp: {temp}")
        if temp >= t_max:
            if mode == "single":
                print("Entering sequential")
                os.system(f"taskset -p -c 2 {p1.pid}")
                os.system(f"taskset -p -c 2 {p2.pid}")
                os.system(f"taskset -p -c 2 {p3.pid}")
                os.system(f"taskset -p -c 2 {p4.pid}")
                time.sleep(3)
            elif mode == "double":
                os.system(f"taskset -p -c 5 {p1.pid}")
                os.system(f"taskset -p -c 5 {p2.pid}")
                os.system(f"taskset -p -c 5 {p3.pid}")
                os.system(f"taskset -p -c 5 {p4.pid}")
        elif temp <= t_min:
                time.sleep(3)
                print("Entering concurrent")
                os.system(f"taskset -p -c 2 {p1.pid}")
                os.system(f"taskset -p -c 3 {p2.pid}")
                os.system(f"taskset -p -c 4 {p3.pid}")
                os.system(f"taskset -p -c 5 {p4.pid}")
        time.sleep(0.5)

    stop_event.set()
    temp_logger.join()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
"""

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    # Start in concurrent mode
    os.system(f"taskset -p -c 4 {p1.pid}")
    os.system(f"taskset -p -c 5 {p2.pid}")
    os.system(f"taskset -p -c 6 {p3.pid}")
    os.system(f"taskset -p -c 7 {p4.pid}")

    hot_exec = False
    cool_exec = False
    while p1.is_alive() or p2.is_alive() or p3.is_alive() or p4.is_alive():
        temps = get_temps()
        max_temp = max(temps)
        avg_temp = mean(temps)

        if avg_temp >= t_max and not hot_exec:
            if mode == "single":
                # Switch to sequential on one core
                os.system(f"taskset -p -c 4 {p1.pid}")
                os.system(f"taskset -p -c 4 {p2.pid}")
                os.system(f"taskset -p -c 4 {p3.pid}")
                os.system(f"taskset -p -c 4 {p4.pid}")
            elif mode == "double":
                # Switch to sequential on two cores
                os.system(f"taskset -p -c 4,5 {p1.pid}")
                os.system(f"taskset -p -c 4,5 {p2.pid}")
                os.system(f"taskset -p -c 6,7 {p3.pid}")
                os.system(f"taskset -p -c 6,7 {p4.pid}")
            hot_exec = True
            cool_exec = False

        if avg_temp <= t_min and not cool_exec:
            # Switch back to concurrent mode
            os.system(f"taskset -p -c 4 {p1.pid}")
            os.system(f"taskset -p -c 5 {p2.pid}")
            os.system(f"taskset -p -c 6 {p3.pid}")
            os.system(f"taskset -p -c 7 {p4.pid}")
            cool_exec = True
            hot_exec = False

        time.sleep(0.04)

    p1.join()
    p2.join()
    p3.join()
    p4.join()

    stop_event.set()  # Signal the temperature logging process to stop
    temp_logger.join()  # Wait for the logging process to finish
"""


def main():
    #scheduler("c")
    scheduler("s")
    #hybrid(t_min=34.2, t_max=42.5, mode="single", dim=400)
    #hybrid(t_min=34.2, t_max=42.5, mode="double", dim=400)

if __name__ == "__main__":
    main()

