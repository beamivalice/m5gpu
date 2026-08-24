# m5gpu — GPU power-cap measurement demo for Apple M5

Credit X: @SorenBGB

This tool sets a power limit on the GPU of an Apple M5 Mac. It then measures
compute speed and power draw at that limit. You can use it to record how GPU
performance changes with power.

The tool uses two public frameworks only: IOKit and Metal. It has no private
APIs, no kernel extensions, and no helper daemons.

## How it works

1. The tool writes a two-key dictionary to the `AGXAccelerator` driver node
   with `IORegistryEntrySetCFProperties`:

   ```
   { "SetMaxGPUAbsolutePower": true, "AbsoluteTarget": <milliwatts> }
   ```

2. The firmware holds the GPU at that power level. It changes the GPU clock
   continuously to stay within ±0.5 % of the target.

3. The tool runs a 4096 × 4096 FP32 matrix multiply on Metal and records:
   - `MaxGPUAbsolutePower` — the limit as accepted by the driver
   - `FilteredGPUPower` — live power draw in milliwatts
   - GPU clock, read from `powermetrics`

## Build

```bash
make
```

You need macOS on an Apple M5 Mac and Xcode Command Line Tools.

## Usage

```bash
m5gpu status                 # show limit and live power (no root)
sudo m5gpu sweep             # measure 5..100 W steps + no-limit (root)
sudo m5gpu sweep 20 40       # measure only the limits you list
sudo m5gpu burn [seconds]    # full-speed burn with live telemetry (root)
```

`sweep` ends by writing `-1000`. This disables the limit and restores stock
GPU behavior.

## Measurement notes

- Each step burns for 7 s. The first 2 s contain a ramp transient. The tool
  discards them and counts only the last 5 s.
- A 4 s cool-down follows every limit change, so each step starts settled.
- Power is sampled after every counted pass. The reported value is the mean.
- Frequency is the median value from `powermetrics` during the burn window.

## Measured results

Apple M5 Max, 40-core GPU, macOS 26.5. Run order: low to high limits, then
no limit.

```
  cap   freq       power     GFLOPS    GF/W
  ---------------------------------------------------------
  5      338 MHz    5.00 W      534    106.9
 10      558 MHz   10.00 W      931     93.0
 15      722 MHz   15.01 W     1218     81.1
 20      890 MHz   19.98 W     1504     75.3
 25      989 MHz   24.99 W     1664     66.6
 30     1075 MHz   30.00 W     1834     61.1
 35     1178 MHz   35.02 W     1988     56.8
 40     1238 MHz   39.99 W     2087     52.2
 50     1328 MHz   50.01 W     2257     45.1
 60     1428 MHz   60.03 W     2421     40.3
 70     1495 MHz   70.04 W     2556     36.5
 80     1546 MHz   80.08 W     2642     33.0
 90     1611 MHz   90.11 W     2745     30.5
100     1620 MHz   95.03 W     2763     29.1
uncap   1620 MHz   96.57 W     2761     28.6
```

How to read this table:

- Below 40 W, the firmware lowers clock and voltage together. Efficiency is
  highest between 8 and 12 W, at about 93 to 107 GFLOPS per watt.
- The GPU hard-caps near 95 W. Limits of 90, 100, and no-limit all converge
  to the same result: 1611 to 1620 MHz, 2745 to 2763 GFLOPS.
- The 5 W step sits on the lowest hardware clock (338 MHz). It gives 107
  GFLOPS per watt. That is 3.6 times better than full power.

## Negative is uncapped

On M5, `0` means "target zero watts". The firmware then parks the GPU at its lowest clock
(338 MHz). It stays there under load, even when thermals are normal.

Write a negative target instead:

```c
agx_set_power_cap(-1000);   // disables the limit; full boost returns
```

| Value written | Registry reads | GPU behavior |
|---|---|---|
| `0` | `0` | parked at 338 MHz (trap) |
| `-1000` | `-1000` | no limit; full boost |
| positive N | N | power held at N watts |

## Requirements and cautions

- `status` works without root. All other commands need root.
- The written value stays active after this tool exits. It also stays active
  for other apps. Always end your session with `sweep`, or write `-1000`.
- These are undocumented driver interfaces. Apple can change them in any
  macOS release. Tested on macOS 26.5 only.

## License

MIT
