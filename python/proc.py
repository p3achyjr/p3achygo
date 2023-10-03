import os, shlex
from subprocess import Popen, PIPE, STDOUT
from threading import Thread

def print_stdout(out):  # pytype : disable=unbound-type-param
  for line in out:
    print(line.rstrip())


def run_proc(cmd: str, env=None):
  if not env:
    env = os.environ

  cmd = shlex.split(cmd)
  proc = Popen(cmd,
               stdin=PIPE,
               stdout=PIPE,
               stderr=STDOUT,
               universal_newlines=True,
               env=env)
  t = Thread(target=print_stdout, args=(proc.stdout,), daemon=True)
  t.start()
  proc.wait()
  return proc.poll()
