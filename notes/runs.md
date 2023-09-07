# v1

<b>Date</b>: 7-14-2023 to 7-24-2023
<b>Num Games</b>: 500,000
<b>Best Model</b>: model_0093
<b>Starting Point</b>: SL Bootstrap

## Self-Elo

#### n=192, k=8, 100 games each

### Elo

| vs         | model_0000 |model_0033 | model_0065 | model_0093 |
|------------|:----------:|:----------:|:----------:|:----------:|
| model_0000 | - | -552.084 +- 26.741 | -798.254 +- 13.558 | -603.860 +- 23.268 |
| model_0033 | 552.084 +- 26.741| - | -301.331 +- 48.953 | -301.331 +- 48.953 |
| model_0065 | 798.254 +- 13.558| 301.331 +- 48.953 | - | -107.538 +- 65.734 |
| model_0093 | 603.860 +- 23.268| 301.331 +- 48.953 | 107.538 +- 65.734 | - |

### Winrate

| vs         | model_0000 |model_0033 | model_0065 | model_0093 |
|------------|:----------:|:----------:|:----------:|:----------:|
| model_0000 | - | <b>4</b> - 96 | <b>1</b> - 99 | <b>3</b> - 97 |
| model_0033 | <b>96</b> - 4 | - | <b>15</b> - 85 | <b>15</b> - 85 |
| model_0065 | <b>99</b> - 1 | <b>85</b> - 15 | - | <b>35</b> - 65 |
| model_0093 | <b>97</b> - 3 | <b>85</b> - 15 | <b>65</b> - 35 | - |

## vs. GnuGo3.8 (Level 10)
#### n=128, k=8, 100 games
*This took way too long, only playing 30 games from now on.*

```
gnugo-l10 v p3achygo-v1 (100/100 games)
board size: 19   komi: 7.5
              wins              black         white       forfeits avg cpu
gnugo-l10        8  8.00%       5  10.00%     3   6.00%          1  186.29
p3achygo-v1     92 92.00%       47 94.00%     45 90.00%          0 1200.39
                                52 52.00%     48 48.00%
```

GnuGo Elo: 1238
Elo: 1662

## vs. Fuego 1.1
#### n=128, k=8, 30 games
```
board size: 19   komi: 7.5
              wins              black         white       avg cpu
fuego           13 43.33%       5  33.33%     8  53.33%   3429.03
p3achygo-v1     17 56.67%       7  46.67%     10 66.67%   1454.42
                                12 40.00%     18 60.00%
```

Fuego Elo: 1506
Elo: 1553

## vs. Pachi 12.82
#### n=128, k=8, 30 games
```
board size: 19   komi: 7.5
              wins              black         white       avg cpu
pachi           19 63.33%       9  60.00%     10 66.67%    767.78
p3achygo-v1     11 36.67%       5  33.33%     6  40.00%   1259.11
                                14 46.67%     16 53.33%
```
Pachi Elo: 1733
Elo: 1638

# v2

<b>Date</b>: 7-25-2023 - 8-01-2023
<b>Num Games</b>: 500,000
<b>Best Model</b>: model_0113
<b>Starting Point</b>: Zero

## Self-Elo

#### Visit Scaling

### Elo

| vs         | n=128, k=16 |n=256, k=16 | n=512, k=16 | n=1024, k=16 |
|------------|:----------:|:----------:|:----------:|:----------:|
| n=128, k=16 | - |  -0.568 +- 0.070 |  |  |
| n=256, k=16 | 0.568 +- 0.070 | - |  |  |
| n=512, k=16 | | | - |  |
| n=1024, k=16 | | | | - |

#### Same Number of Visits on Chosen Child

### Elo

| vs         | n=205, k=8 | n=256, k=16 | n=309, k=32 |
|------------|:----------:|:----------:|:----------:|
| n=205, k=8 | - |  -0.568 +- 0.070 |  -0.560 +- 0.097 |  
| n=256, k=16 | 0.568 +- 0.070 | - |  -0.550 +- 0.098| 
| n=309, k=32 | 0.560 +- 0.097 | 0.550 +- 0.098| - |  

## vs. Pachi 12.82
#### n=256, k=8, 30 games
```
pachi v p3achygo-v2 (30/30 games)
board size: 19   komi: 7.5
              wins              black         white        avg cpu
pachi            2  6.67%       0   0.00%     2   13.33%    778.27
p3achygo-v2     28 93.33%       13 86.67%     15 100.00%   2270.71
                                13 43.33%     17  56.67%
```
Pachi Elo: 1733
Elo: 2191

## vs. Leela 066
`leelaz -g --noponder -p 1000 -w 066.gz`
`p3achygo --model_path=/tmp/p3achygo/models/model_0113/_trt --n=256`
```
leela v p3achygo-v2 (100/100 games)
board size: 19   komi: 7.5
              wins              black         white       forfeits avg cpu
leela           44 44.00%       19 38.00%     25 50.00%          1   44.03
p3achygo-v2     56 56.00%       25 50.00%     31 62.00%          0  141.38
                                44 44.00%     56 56.00%
```
Leela Elo: 2331
Elo: 2393

## vs. Leela 071
`leelaz -g --noponder -p 1000 -w 071.gz`
`p3achygo --model_path=/tmp/p3achygo/models/model_0113/_trt --n=256`
```
leela v p3achygo-v2 (100/100 games)
board size: 19   komi: 7.5
              wins              black         white       avg cpu
leela           59 59.00%       30 60.00%     29 58.00%     32.36
p3achygo-v2     41 41.00%       21 42.00%     20 40.00%    139.96
                                51 51.00%     49 49.00%
```

Leela Elo: 2485
Elo: 2422

## vs. Leela 076
`leelaz -g --noponder -p 1000 -w 076.gz`
`p3achygo --model_path=/tmp/p3achygo/models/model_0113/_trt --n=256`
```
leela v p3achygo-v2 (30/30 games)
board size: 19   komi: 7.5
              wins              black         white       forfeits avg cpu
leela           18 60.00%       10 66.67%     8  53.33%          1   30.94
p3achygo-v2     12 40.00%       7  46.67%     5  33.33%          0  135.37
                                17 56.67%     13 43.33%
```

Leela Elo: 2569
Elo: 2499

# v2 500-1000k

<b>Date</b>: 8-04-2023 - 8-10-2023
<b>Num Games</b>: 250,000
<b>Best Model</b>: model_0141
<b>Starting Point</b>: v2

## Self-Elo

### vs v2-model_0113

+200 Elo

## vs. Leela 071
`leelaz -g --noponder -p 1600 -w 071.gz`
`p3achygo --model_path=/tmp/p3achygo/models/model_0113/_trt --n=256`

```
leela v p3achygo-v2 (100/100 games)
board size: 19   komi: 7.5
              wins              black         white       avg cpu
leela           56 56.00%       29 58.00%     27 54.00%     55.54
p3achygo-v2     44 44.00%       23 46.00%     21 42.00%    253.79
                                52 52.00%     48 48.00%
```

## vs. Leela 076
`leelaz -g --noponder -p 1600 -w 076.gz`
`p3achygo --model_path=/tmp/p3achygo/models/model_0113/_trt --n=256`
```
leela v p3achygo-v2 (100/100 games)
board size: 19   komi: 7.5
              wins              black         white       avg cpu
leela           67 67.00%       32 64.00%     35 70.00%     52.69
p3achygo-v2     33 33.00%       15 30.00%     18 36.00%    239.24
                                47 47.00%     53 53.00%
```
