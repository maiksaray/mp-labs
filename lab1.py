from multiprocessing import cpu_count
import jinja2
from datetime import datetime
import subprocess as sp
from os import path
from pathlib import Path
import random


def get_thread_numbers(max_number):
    number = 1
    while number <= max_number:
        yield number
        number <<= 1


def size_string(data_sizes):
    return "x".join(map(str, data_sizes))


def schedule_string(schedule):
    return f"{schedule['schedule_type']}" + \
           (f"-chunk{schedule['chunk_size']}" if 'chunk_size' in schedule else "")


class Runner:

    def __init__(self, template_path="templates", compiler_path=None, compiler_exec="cl", data_path="data",
                 precompile_path=None):
        fs_loader = jinja2.FileSystemLoader(searchpath=template_path)
        self.env = jinja2.Environment(loader=fs_loader)
        self.precompile_path = precompile_path
        self.compiler_path = compiler_path
        self.compiler_exec = compiler_exec
        self.data_path = data_path

    def generate_matrix(self, n, m):
        return [[random.random() for x in range(m)] for y in range(n)]

    def write_matrix(self, matrix, file):
        file.write(f"{len(matrix)} {len(matrix[0]) if matrix else ''}\n")
        for row in matrix:
            for item in row:
                file.write(f"{item:.2f} ")
            file.write("\n")

    def generate_data_dir(self, a_path, b_path, data_sizes):
        print(f"Generating data for {size_string(data_sizes)}")
        n = data_sizes[0]
        m = data_sizes[1]
        k = data_sizes[2]
        Path(a_path).parent.mkdir(parents=True, exist_ok=True)
        with open(a_path, "w+") as a_file:
            self.write_matrix(self.generate_matrix(n, m), a_file)
        Path(b_path).parent.mkdir(parents=True, exist_ok=True)
        with open(b_path, "w+") as b_file:
            self.write_matrix(self.generate_matrix(m, k), b_file)

    def perform_run(self, loop, threads, sizes, schedule):
        src_dir, src_name = self.render_src(loop, schedule, threads)

        self.compile_src(src_dir, src_name)

        print("Compiled, doing stuff...")
        results = dict()

        start = datetime.now()
        for size_spec in sizes:
            duration = self.perform_single_run(path.join(src_dir, src_name.replace(".cpp", ".exe")), size_spec)
            results.update({size_string(size_spec): duration})
        end = datetime.now()
        print(end - start)
        return results

    def compile_src(self, src_dir, src_name):
        print(f"Compiling rendered {src_name}")
        cl_proc = sp.Popen([path.join(self.precompile_path, "VsDevCmd.bat"), "&",
                            path.join(self.compiler_path, self.compiler_exec), "/EHsc", src_name],
                           stdout=sp.PIPE, stderr=sp.PIPE, shell=True,
                           cwd=src_dir)
        stdout, stderr = cl_proc.communicate()
        # if cl_proc.returncode

    def render_src(self, loop, schedule, threads):
        threads = str(threads)
        params = {"threads": threads,
                  "schedule": schedule}
        omp_line = self.env.get_template("lab1.omp.j2").render(**schedule)
        params.update({
            loop: omp_line
        })
        rendered = self.env.get_template("lab1.cpp.j2").render(params)
        src_dir = f"run/{loop}/{threads}"
        Path(src_dir).mkdir(parents=True, exist_ok=True)
        src_name = f"{schedule_string(schedule)}.cpp"
        with open(path.join(src_dir, src_name), "w+") as src:
            src.write(rendered)
        return src_dir, src_name

    def perform_single_run(self, exec, data_sizes):
        run_data_dir = path.join(self.data_path, size_string(data_sizes))
        a_path = path.join(run_data_dir, "a")
        b_path = path.join(run_data_dir, "b")

        if not path.exists(run_data_dir):
            self.generate_data_dir(a_path, b_path, data_sizes)

        print(f"starting {size_string(data_sizes)}")
        process = sp.Popen([exec, a_path, b_path], stdout=sp.PIPE, stderr=sp.PIPE)
        stdout, stderr = process.communicate()
        if stderr:
            print(stderr.decode("utf-8"))
        return stdout.decode("utf-8").strip()


loops = ["outer", "inner"]
threads = list(get_thread_numbers(cpu_count()))
schedules = [
    {"schedule_type": "dynamic"},
    {"schedule_type": "static"},
    {"schedule_type": "guided"},
    {"schedule_type": "dynamic",
     "chunk-size": "100"},
    {"schedule_type": "static",
     "chunk-size": "100"},
]
sizes = [[500, 500, 500],
         [800, 800, 800],
         [8000, 80, 800],
         [64000, 8, 800]
         ]

runner = Runner(
    compiler_path="C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.24.28314\\bin\\Hostx64\\x86",
    precompile_path="C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\Common7\\Tools")

total_results = list()

for paralleled_loop in loops:
    for thread_number in threads:
        for schedule in schedules:
            result = runner.perform_run(paralleled_loop, thread_number, sizes, schedule)
            for size in result:
                total_results.append(
                    {"loop": paralleled_loop,
                     "threads": thread_number,
                     "schedule": schedule_string(schedule),
                     "size": size,
                     "duration": result[size]
                     })
            print(f"done with schedule {schedule_string(schedule)}")
        print(f"done with thread {thread_number}")


if not total_results:
    exit(-1)

import csv
with open("lab1_results.csv", "w+") as out:
    writer = csv.DictWriter(out, total_results[0].keys(), delimiter=";")
    writer.writeheader()
    writer.writerows(total_results)

import tabulate

print(tabulate.tabulate(total_results))
