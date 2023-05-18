from pathlib import Path


def main():
  path = Path('/tmp/test_ds')
  for i in range(100):
    with open(path.joinpath(f'f{i}.txt').absolute(), 'w') as f:
      for j in range(100):
        f.write(f'f{i} m{j}\n')


if __name__ == '__main__':
  main()
