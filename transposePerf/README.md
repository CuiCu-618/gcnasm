**Transpose/Copy microbenchmark (ROCm/HIP).**

Compares baseline copy, naive transpose, coalesced tile transpose, and bank-conflict-free variants on AMD GPUs. Optional “XCD swizzle” remaps block IDs to spread work across XCDs.

**Build**

```bash
# using the optimized Makefile you have:
make BUILD=Release GPU_ARCHS="gfx942"
```

**Run**

```bash
./transposePerf [width] [height] 
# example:
./transposePerf 2048 2048
```

**Output**
```
> Device:        AMD Instinct MI300A
> gcnArchName:   gfx942:sramecc+:xnack-
> Number of CUs: 228

Matrix 1024x1024, tiles 16x16, tile 64x64, block 64x16

Variant                        |    BW (GB/s) |  Time (ms) |     Elements |     WG
-------------------------------+--------------+------------+--------------+-------
copy (baseline)                |     1540.553 |     0.0051 |      1048576 |   1024
copy + xcd swizzle             |     1325.132 |     0.0059 |      1048576 |   1024
copy + xcd swizzle (group)     |     1316.468 |     0.0059 |      1048576 |   1024
transpose naive                |       84.315 |     0.0927 |      1048576 |   1024
transpose naive + swizzle      |       84.170 |     0.0928 |      1048576 |   1024
transpose coalesced            |     1069.729 |     0.0073 |      1048576 |   1024
transpose coalesced + swzl     |      984.481 |     0.0079 |      1048576 |   1024
transpose no-bank              |     1443.864 |     0.0054 |      1048576 |   1024
transpose no-bank + swzl       |     1324.952 |     0.0059 |      1048576 |   1024
transpose diagnoal             |     1442.158 |     0.0054 |      1048576 |   1024
transpose diagnoal + swzl      |     1370.797 |     0.0057 |      1048576 |   1024
transpose swizzle + swzl       |      979.249 |     0.0080 |      1048576 |   1024
-------------------------------+--------------+------------+--------------+-------
Test passed


Matrix 2048x2048, tiles 32x32, tile 64x64, block 64x16

Variant                        |    BW (GB/s) |  Time (ms) |     Elements |     WG
-------------------------------+--------------+------------+--------------+-------
copy (baseline)                |       73.916 |     0.4228 |      4194304 |   1024
copy + xcd swizzle             |     2648.741 |     0.0118 |      4194304 |   1024
copy + xcd swizzle (group)     |     2768.118 |     0.0113 |      4194304 |   1024
transpose naive                |       84.231 |     0.3710 |      4194304 |   1024
transpose naive + swizzle      |      111.998 |     0.2790 |      4194304 |   1024
transpose coalesced            |      114.816 |     0.2722 |      4194304 |   1024
transpose coalesced + swzl     |      376.307 |     0.0830 |      4194304 |   1024
transpose no-bank              |      114.408 |     0.2731 |      4194304 |   1024
transpose no-bank + swzl       |      375.818 |     0.0832 |      4194304 |   1024
transpose diagnoal             |      110.986 |     0.2816 |      4194304 |   1024
transpose diagnoal + swzl      |      233.139 |     0.1340 |      4194304 |   1024
transpose swizzle + swzl       |     1014.168 |     0.0308 |      4194304 |   1024
-------------------------------+--------------+------------+--------------+-------
Test passed


Matrix 2048x2048, tiles 32x32, tile 64x64, block 64x8

Variant                        |    BW (GB/s) |  Time (ms) |     Elements |     WG
-------------------------------+--------------+------------+--------------+-------
copy (baseline)                |       73.337 |     0.4261 |      4194304 |    512
copy + xcd swizzle             |     2705.141 |     0.0116 |      4194304 |    512
copy + xcd swizzle (group)     |     2739.380 |     0.0114 |      4194304 |    512
transpose naive                |       75.277 |     0.4151 |      4194304 |    512
transpose naive + swizzle      |       96.946 |     0.3223 |      4194304 |    512
transpose coalesced            |      115.323 |     0.2710 |      4194304 |    512
transpose coalesced + swzl     |      373.217 |     0.0837 |      4194304 |    512
transpose no-bank              |      114.082 |     0.2739 |      4194304 |    512
transpose no-bank + swzl       |      373.951 |     0.0836 |      4194304 |    512
transpose diagnoal             |      105.662 |     0.2958 |      4194304 |    512
transpose diagnoal + swzl      |      239.805 |     0.1303 |      4194304 |    512
transpose swizzle + swzl       |      911.891 |     0.0343 |      4194304 |    512
-------------------------------+--------------+------------+--------------+-------
Test passed


Matrix 4096x4096, tiles 64x64, tile 64x64, block 64x16

Variant                        |    BW (GB/s) |  Time (ms) |     Elements |     WG
-------------------------------+--------------+------------+--------------+-------
copy (baseline)                |       76.044 |     1.6438 |     16777216 |   1024
copy + xcd swizzle             |      383.625 |     0.3258 |     16777216 |   1024
copy + xcd swizzle (group)     |      367.266 |     0.3404 |     16777216 |   1024
transpose naive                |        1.290 |    96.9331 |     16777216 |   1024
transpose naive + swizzle      |        1.169 |   106.9399 |     16777216 |   1024
transpose coalesced            |       51.391 |     2.4323 |     16777216 |   1024
transpose coalesced + swzl     |       68.102 |     1.8355 |     16777216 |   1024
transpose no-bank              |       51.514 |     2.4265 |     16777216 |   1024
transpose no-bank + swzl       |       68.027 |     1.8375 |     16777216 |   1024
transpose diagnoal             |       79.044 |     1.5814 |     16777216 |   1024
transpose diagnoal + swzl      |       45.365 |     2.7554 |     16777216 |   1024
transpose swizzle + swzl       |      192.113 |     0.6507 |     16777216 |   1024
-------------------------------+--------------+------------+--------------+-------
Test passed
```