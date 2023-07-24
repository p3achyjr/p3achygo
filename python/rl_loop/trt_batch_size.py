import multiprocessing

THREAD_CPU_RATIO = 6


def trt_batch_size() -> int:
  est = multiprocessing.cpu_count() * THREAD_CPU_RATIO

  # round up to next larget power of 16
  return (est // 16 + 1) * 16
