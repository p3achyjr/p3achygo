# Gumbel Benchmarks 7-03-2023

Info:

```
Run on (4 X 2200 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x2)
  L1 Instruction 32 KiB (x2)
  L2 Unified 256 KiB (x2)
  L3 Unified 56320 KiB (x1)
```

## Base

```
Load Average: 11.39, 11.66, 10.90
----------------------------------------------------------
Benchmark                Time             CPU   Iterations
----------------------------------------------------------
BM_Gumbel/1/1        51711 ns        51732 ns        10000
BM_Gumbel/8/2       162215 ns       162230 ns         3924
BM_Gumbel/32/4      997007 ns       997149 ns          705
BM_Gumbel/48/8     1531176 ns      1531323 ns          477
BM_Gumbel/64/4     3495129 ns      3495105 ns          191
BM_Gumbel/128/4   14761414 ns     14712092 ns           57
```

```
Load Average: 8.94, 11.10, 10.73
----------------------------------------------------------
Benchmark                Time             CPU   Iterations
----------------------------------------------------------
BM_Gumbel/1/1        48830 ns        48703 ns        14516
BM_Gumbel/8/2       153557 ns       153577 ns         3810
BM_Gumbel/32/4      974300 ns       974481 ns          706
BM_Gumbel/48/8     1470271 ns      1470422 ns          484
BM_Gumbel/64/4     3454439 ns      3454386 ns          205
BM_Gumbel/128/4   12694795 ns     12694125 ns           58

```

## SSE Softmax

```
Load Average: 13.61, 11.81, 11.00
----------------------------------------------------------
Benchmark                Time             CPU   Iterations
----------------------------------------------------------
BM_Gumbel/1/1        47864 ns        47882 ns        14772
BM_Gumbel/8/2       130686 ns       130699 ns         5315
BM_Gumbel/32/4      763564 ns       763715 ns          910
BM_Gumbel/48/8     1139889 ns      1140005 ns          617
BM_Gumbel/64/4     2655522 ns      2655587 ns          260
BM_Gumbel/128/4   10219704 ns     10219018 ns           68
```

```
Load Average: 11.59, 11.44, 10.89
----------------------------------------------------------
Benchmark                Time             CPU   Iterations
----------------------------------------------------------
BM_Gumbel/1/1        48237 ns        48250 ns        14395
BM_Gumbel/8/2       130308 ns       130313 ns         5291
BM_Gumbel/32/4      802552 ns       802732 ns          893
BM_Gumbel/48/8     1199269 ns      1199414 ns          606
BM_Gumbel/64/4     2854670 ns      2838920 ns          260
BM_Gumbel/128/4   10100666 ns     10099895 ns           72
```

## AVX Softmax

```
Load Average: 10.94, 11.38, 11.44
----------------------------------------------------------
Benchmark                Time             CPU   Iterations
----------------------------------------------------------
BM_Gumbel/1/1        48244 ns        48220 ns        14362
BM_Gumbel/8/2       146425 ns       146444 ns         4866
BM_Gumbel/32/4      818913 ns       819110 ns          867
BM_Gumbel/48/8     1230915 ns      1231037 ns          600
BM_Gumbel/64/4     2924662 ns      2924651 ns          250
BM_Gumbel/128/4   10877289 ns     10876696 ns           70
```

```
Load Average: 12.84, 11.75, 11.56
----------------------------------------------------------
Benchmark                Time             CPU   Iterations
----------------------------------------------------------
BM_Gumbel/1/1        48633 ns        48665 ns        11797
BM_Gumbel/8/2       141356 ns       141383 ns         5013
BM_Gumbel/32/4      832092 ns       832284 ns          868
BM_Gumbel/48/8     1200941 ns      1201148 ns          581
BM_Gumbel/64/4     2965452 ns      2932516 ns          257
BM_Gumbel/128/4   10396101 ns     10395537 ns           66
```

## Inline Tree Functions

```
Load Average: 16.72, 13.32, 11.86
----------------------------------------------------------
Benchmark                Time             CPU   Iterations
----------------------------------------------------------
BM_Gumbel/1/1        51364 ns        51384 ns        10000
BM_Gumbel/8/2       100900 ns       100925 ns         6710
BM_Gumbel/32/4      454049 ns       454277 ns         1576
BM_Gumbel/48/8      755509 ns       755791 ns         1014
BM_Gumbel/64/4     1416342 ns      1416591 ns          507
BM_Gumbel/128/4    5022506 ns      5022394 ns          145
```

```
Load Average: 19.68, 13.76, 11.98
----------------------------------------------------------
Benchmark                Time             CPU   Iterations
----------------------------------------------------------
BM_Gumbel/1/1        48948 ns        48956 ns        14831
BM_Gumbel/8/2        99146 ns        99155 ns         6747
BM_Gumbel/32/4      454944 ns       455129 ns         1612
BM_Gumbel/48/8      647135 ns       647340 ns         1003
BM_Gumbel/64/4     1412684 ns      1412935 ns          522
BM_Gumbel/128/4    4795869 ns      4795923 ns          147
```
