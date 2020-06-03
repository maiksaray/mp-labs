import numpy as np
from pathlib import Path
import subprocess as sp


def rank(A):
    return np.linalg.matrix_rank(A)


def generate_data_dir(size_data_path, size):
    size_data_path.mkdir(parents=True, exist_ok=True)
    A = np.around(np.random.rand(size, size), 5)
    b = np.around(np.random.rand(size), 5)
    x = np.ones(size)

    row_sums = np.sum(A, axis=1)
    d = np.diagflat(row_sums + np.random.rand(size))  # a little trick to make sure that diag is big enough
    A += d
    while rank(A) != rank(np.c_[A, b]):
        A = np.random.rand(size, size)
        b = np.random.rand(size)
        row_sums = np.sum(A, axis=1)
        A += np.diagflat(row_sums + np.random.rand(size))

    A_path = size_data_path.joinpath("A")
    np.savetxt(A_path, A)
    b_path = size_data_path.joinpath("b")
    np.savetxt(b_path, b)
    x_path = size_data_path.joinpath("x")
    np.savetxt(x_path, x)
    return A_path, b_path, x_path


def run_reference(A_path, b_path, x_path):
    ref_run = sp.Popen([f"python lab2_sequential.py",
                        f"--A {A_path}",
                        f"--b {b_path}",
                        f"--x {x_path}"], stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = ref_run.communicate()
    ref_run_results = out.decode()
    if "failed" not in ref_run_results:
        return ref_run_results
    print(f"Reference run failed")


def run_test(A_path, b_path, x_path, threads):
    test_run = sp.Popen([f"mpiexec -n {threads} python -m mpi4py lab2.py",
                         f"--A {A_path}",
                         f"--b {b_path}",
                         f"--x {x_path}"], stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = test_run.communicate()
    test_run_results = out.decode()
    if "failed" not in test_run_results:
        return test_run_results
    print(f"{threads} Threads test failed")


def avg(nums):
    nums.remove(max(nums))
    nums.remove(min(nums))
    return sum(nums) / len(nums)


data_dir = "data"
data_path = Path(data_dir)

NUMBER_OF_RUNS = 5
sizes = [80, 160, 320, 800, 1600]
threads = [2, 4, 8]

for size in sizes:

    size_data_path = data_path.joinpath(str(size))
    if not size_data_path.exists():
        generate_data_dir(size_data_path, size)

    reference_results = [run_reference(size_data_path.joinpath("A"),
                                       size_data_path.joinpath("b"),
                                       size_data_path.joinpath("x"))
                         for _ in range(NUMBER_OF_RUNS)]

    # we take a list of strings with numbers divided by spaces
    # and get two lists of first numbers and second numbers
    # ["a b", "c d"] -> [a, c], [b,d]
    man_results, np_results = map(lambda t: list(map(float, t)),
                                  zip(*map(str.split, reference_results)))

    if reference_results:
        print(f"size {size}: manual: {avg(man_results)}; np: {avg(np_results)}")
    for thread in threads:
        test_result = [run_test(size_data_path.joinpath("A"),
                                size_data_path.joinpath("b"),
                                size_data_path.joinpath("x"),
                                thread)
                       for _ in range(NUMBER_OF_RUNS)]
        test_numbers = list(map(float, test_result))
        print(f"threads {thread}: {avg(test_numbers)}")
print("finished")
