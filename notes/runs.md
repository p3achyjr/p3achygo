# v1

<b>Date</b>: 7-14-2023 to 7-24-2023
<b>Num Games</b>: 500,000
<b>Best Model</b>: model_0093
<b>From SL Bootstrap?</b>: Yes

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
