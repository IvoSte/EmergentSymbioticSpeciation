import threading
from multiprocessing import Queue, Process, cpu_count, Pool
import multiprocessing
import dill
from tqdm import tqdm
from .logger import log

# Change multiprocessing pickler to dill (https://stackoverflow.com/a/69253561/9302146)
# dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
# multiprocessing.reduction.ForkingPickler = dill.Pickler
# multiprocessing.reduction.dump = dill.dump
# multiprocessing.queues._ForkingPickler = dill.Pickler


class ThreadPool:
    def __init__(self):
        self.max_num_threads = cpu_count()
        self.threads = []
        self.queue = Queue()

    def add_task(self, function, *args, **kwargs):
        thread = threading.Thread(target=function, args=args, kwargs=kwargs)
        self.threads.append(thread)
        thread.start()

    def wait_completion(self):
        for thread in self.threads:
            thread.join()

    def get_result(self):
        return self.queue.get()


class MultiProcessor:
    def __init__(self, max_num_processes: int = None):
        self.max_num_processes = (
            cpu_count()
            if max_num_processes == None
            else min(max_num_processes, cpu_count())
        )
        log.info(f"Using {self.max_num_processes} processes / CPU cores")
        self.processes = []
        self.queue = Queue()

    def add_task(self, function, *args, **kwargs):
        process = Process(target=function, args=args, kwargs=kwargs)
        self.processes.append(process)
        process.start()

    def add_task_dill(self, function, *args, **kwargs):
        process = DillProcess(target=function, args=args, kwargs=kwargs)
        self.processes.append(process)
        process.start()

    def run_pool(self, function, input_list):
        with Pool(processes=self.max_num_processes) as pool:
            results = tqdm(
                pool.imap_unordered(function, input_list),
                total=len(input_list),
            )
            for result in results:
                self.queue.put(result)

    # def run_map(self, function, input_list):
    #     #     with mp.Pool(processes=MAX_WORKERS) as pool:
    #     #     results = tqdm(
    #     #         pool.imap_unordered(foo, inputs, chunksize=CHUNK_SIZE),
    #     #         total=len(inputs),
    #     #     )  # 'total' is redundant here but can be useful
    #     #     # when the size of the iterable is unobvious
    #     with multiprocessing.Pool(processes=self.max_num_processes) as pool:
    #         results = tqdm(
    #             pool.imap_unordered(function, input_list),
    #             total=len(input_list),
    #         )
    #         for result in results:
    #             self.queue.put(result)

    # def add_task_progress_bar(self, function, *args, **kwargs):
    #     process = Process(target=function, args=args, kwargs=kwargs)
    #     self.processes.append(process)

    # def start_progress_bar(self):
    #     pass

    def wait_completion(self):
        for process in self.processes:
            process.join()
        for process in self.processes:
            process.close()

    def wait_completion_with_pbar(self):
        print("Warning. Fixing progress bars takes more time than you expect.")
        for process in tqdm(self.processes, colour="green"):
            process.join()

    def get_result(self):
        return self.queue.get()


class DillProcess(Process):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target = dill.dumps(
            self._target
        )  # Save the target function as bytes, using dill

    def run(self):
        if self._target:
            self._target = dill.loads(
                self._target
            )  # Unpickle the target function before executing
            self._target(*self._args, **self._kwargs)  # Execute the target function
