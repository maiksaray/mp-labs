import numpy as np
from pathlib import Path
import subprocess as sp


def run(ver, data_path, threads):
    process = sp.Popen([f"mpiexec -n {threads} python -m mpi4py lab3_{ver}.py",
                        f"--array {data_path}"
                        ], stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = process.communicate()
    results = out.decode()
    return results.splitlines()[-1]


def generate_data(path, size):
    arr = np.random.randint(size * 2, size=size)
    np.savetxt(path, arr.astype(int))


def avg(nums):
    nums.remove(max(nums))
    nums.remove(min(nums))
    return sum(nums) / len(nums)


data_dir = "data"
number_of_runs = 5
sizes = [80_000, 160_000, 800_000, 1_600_000, 8_000_000, 16_000_000]
threads = [2, 4, 8]

for size in sizes:

    size_data_path = Path(data_dir).joinpath(str(size))
    if not size_data_path.exists():
        generate_data(size_data_path, size)
    for thread in threads:
        list_results = [run("list", size_data_path, thread)
                        for _ in range(number_of_runs)]
        avg_for_list = avg(list(map(float, list_results)))
        print(f"{size}|{thread} list:{avg_for_list}ms")

        np_results = [run("np", size_data_path, thread)
                      for _ in range(number_of_runs)]
        avg_for_np = avg(list(map(float, np_results)))
        print(f"{size}|{thread} np:{avg_for_np}ms")

print("finished")
