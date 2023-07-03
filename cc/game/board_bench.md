## No Optimizations

Run on (4 X 2200 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x2)
  L1 Instruction 32 KiB (x2)
  L2 Unified 256 KiB (x2)
  L3 Unified 56320 KiB (x1)
Load Average: 10.02, 9.29, 9.23
-------------------------------------------------------------------
Benchmark                         Time             CPU   Iterations
-------------------------------------------------------------------
BM_BasicSequence               2996 ns         2995 ns       242169
BM_PonnukiSequence             3069 ns         3069 ns       221342
BM_SanSanSequence              7468 ns         7456 ns       100861
BM_BigChainJoinsSequence       7401 ns         7400 ns        96796
BM_LadderSequence              7287 ns         7286 ns        97501

Run on (4 X 2200 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x2)
  L1 Instruction 32 KiB (x2)
  L2 Unified 256 KiB (x2)
  L3 Unified 56320 KiB (x1)
Load Average: 12.63, 10.21, 9.56
-------------------------------------------------------------------
Benchmark                         Time             CPU   Iterations
-------------------------------------------------------------------
BM_BasicSequence               3001 ns         2983 ns       242369
BM_PonnukiSequence             3094 ns         3094 ns       225959
BM_SanSanSequence              7132 ns         7131 ns        98827
BM_BigChainJoinsSequence       7344 ns         7343 ns        95962
BM_LadderSequence              7291 ns         7290 ns        96290

## Inline Hints (No Difference)

Run on (4 X 2200 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x2)
  L1 Instruction 32 KiB (x2)
  L2 Unified 256 KiB (x2)
  L3 Unified 56320 KiB (x1)
Load Average: 11.74, 11.02, 10.03
-------------------------------------------------------------------
Benchmark                         Time             CPU   Iterations
-------------------------------------------------------------------
BM_BasicSequence               2919 ns         2919 ns       239268
BM_PonnukiSequence             3068 ns         3068 ns       230323
BM_SanSanSequence              7373 ns         7371 ns        97588
BM_BigChainJoinsSequence       7628 ns         7627 ns        82411
BM_LadderSequence              7277 ns         7276 ns        93660

Run on (4 X 2200 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x2)
  L1 Instruction 32 KiB (x2)
  L2 Unified 256 KiB (x2)
  L3 Unified 56320 KiB (x1)
Load Average: 10.88, 10.85, 9.98
-------------------------------------------------------------------
Benchmark                         Time             CPU   Iterations
-------------------------------------------------------------------
BM_BasicSequence               3016 ns         3009 ns       234398
BM_PonnukiSequence             3381 ns         3365 ns       224033
BM_SanSanSequence              7081 ns         7080 ns        98309
BM_BigChainJoinsSequence       7308 ns         7307 ns        90596
BM_LadderSequence              7322 ns         7320 ns        85827

## Calculate Adjacent Indices Up-front (~10% Savings)

Run on (4 X 2200 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x2)
  L1 Instruction 32 KiB (x2)
  L2 Unified 256 KiB (x2)
  L3 Unified 56320 KiB (x1)
Load Average: 17.15, 13.96, 11.73
-------------------------------------------------------------------
Benchmark                         Time             CPU   Iterations
-------------------------------------------------------------------
BM_BasicSequence               2710 ns         2709 ns       242813
BM_PonnukiSequence             2963 ns         2963 ns       238320
BM_SanSanSequence              6930 ns         6828 ns       108281
BM_BigChainJoinsSequence       6533 ns         6532 ns       105756
BM_LadderSequence              6604 ns         6604 ns       108850

Run on (4 X 2200 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x2)
  L1 Instruction 32 KiB (x2)
  L2 Unified 256 KiB (x2)
  L3 Unified 56320 KiB (x1)
Load Average: 15.86, 13.74, 11.67
-------------------------------------------------------------------
Benchmark                         Time             CPU   Iterations
-------------------------------------------------------------------
BM_BasicSequence               2724 ns         2723 ns       261395
BM_PonnukiSequence             2889 ns         2889 ns       247328
BM_SanSanSequence              6419 ns         6418 ns       111772
BM_BigChainJoinsSequence       7137 ns         7104 ns       105828
BM_LadderSequence              6404 ns         6403 ns       109197
