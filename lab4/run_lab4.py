import numpy as np
from pathlib import Path
import subprocess as sp


def run_ref(ver, data_path):
    process = sp.Popen([f"python lab4_{ver}.py",
                        f"--graph {data_path}"
                        ], stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = process.communicate()
    results = out.decode()
    return results.splitlines()[-1]


def run(ver, data_path, threads):
    process = sp.Popen([f"mpiexec -n {threads} python -m mpi4py lab4_{ver}.py",
                        f"--graph {data_path}"
                        ], stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = process.communicate()
    results = out.decode()
    return results.splitlines()[-1]


def generate_data(path, size):
    g = np.random.randint(size * 2, size=(size, size))
    g -= np.diagflat(g.diagonal())
    mask = np.random.choice([False, True], size=(size, size))
    g = np.ma.masked_array(g, mask).filled(0)
    iu = np.triu_indices(size, 1)
    il = (iu[1], iu[0])
    g[il] = g[iu]
    if not np.any(g.sum(axis=0)):
        print("Wow, we got zero row randomly!")
    np.savetxt(path, g.astype(int))


def avg(nums):
    nums.remove(max(nums))
    nums.remove(min(nums))
    return sum(nums) / len(nums)


data_dir = "data"
number_of_runs = 5
sizes = [800, 1_600, 8_000, 16_000, 80_000, 160_000]
threads = [2, 4, 8]

for size in sizes:

    size_data_path = Path(data_dir).joinpath(str(size))

    if not size_data_path.exists():
        generate_data(size_data_path, size)

    sequential_results = [run_ref("sequential", size_data_path)
                          for _ in range(number_of_runs)]
    avg_for_seq = avg(list(map(float, sequential_results)))
    print(f"{size} sequential:{avg_for_seq}ms")

    for thread in threads:
        parallel_results = [run("parallel", size_data_path, thread)
                            for _ in range(number_of_runs)]
        parallel_tuples = list(map(str.split, parallel_results))
        lst_res, np_res = zip(*parallel_tuples)

        avg_for_np = avg(list(map(float, np_res)))
        print(f"{size}|{thread} np:{avg_for_np}ms")

        avg_for_list = avg(list(map(float, lst_res)))
        print(f"{size}|{thread} list:{avg_for_list}ms")
