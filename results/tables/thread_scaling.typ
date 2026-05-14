#figure(
  text(size: 9pt,
    table(
      columns: (auto, auto, auto, auto, auto),
      align: (center, center, center, center, center),
      
      table.header(
        [*Threads*],
        [*TTFT short (s)*],
        [*TPOT (ms)*],
        [*Throughput\ (tok/s)*],
        [*Peak Mem.\ (MB)*],
      ),
      [4],  [5.98 ± 1.77],  [647.12 ± 0.26], [1.56 ± 0.00],  [5856.4],
      [8],  [3.02 ± 0.89],  [329.12 ± 0.18], [3.07 ± 0.00],  [5950.4],
      [16],  [1.54 ± 0.44],  [170.52 ± 0.08], [5.92 ± 0.00],  [6034.3],
      table.cell(fill: luma(220))[*32*], table.cell(fill: luma(220))[*0.78 ± 0.22*], table.cell(fill: luma(220))[*90.41 ± 0.05*], table.cell(fill: luma(220))[*11.17 ± 0.00*], table.cell(fill: luma(220))[*5838.8*],
      [48],  [0.62 ± 0.16],  [88.99 ± 8.83], [11.65 ± 1.13],  [6030.6],
    )
  ),
  caption: [Resultados obtidos pelo modelo Llama-3.1-8B com diferentes threads.],
) <tab:thread-scaling>
