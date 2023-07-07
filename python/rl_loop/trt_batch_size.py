import multiprocessing

THREAD_CPU_RATIO = 6


def trt_batch_size() -> int:
  return multiprocessing.cpu_count() * THREAD_CPU_RATIO
