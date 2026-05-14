#figure(
  text(size: 9pt,
    table(
      columns: (auto, auto, auto, auto, auto, auto),
      align: (left, center, center, center, center, center),

      table.header(
        [*Quantização*],
        [*Category*],
        [*TTFT (s)*],
        [*TPOT (ms)*],
        [*Throughput\ (tok/s)*],
        [*Peak Mem.\ (MB)*],
      ),

      // Q4_K_M
      table.cell(rowspan: 3)[Q4\_K\_M],
      [Short],  [0.78 ± 0.22],  [90.41 ± 0.05], [11.17 ± 0.00], [5838.8],
      [Medium],  [2.73 ± 1.25],  [90.79 ± 0.19], [11.12 ± 0.02], [5972.7],
      [Long],  [25.04 ± 5.64],  [95.74 ± 1.22], [10.55 ± 0.14], [6072.3],

      // Q8_0
      table.cell(rowspan: 3)[Q8\_0],
      [Short],  [0.33 ± 0.08],  [84.25 ± 0.15], [11.99 ± 0.02], [9490.0],
      [Medium],  [1.10 ± 0.69],  [84.47 ± 0.23], [11.96 ± 0.03], [9617.7],
      [Long],  [12.00 ± 2.87],  [89.33 ± 1.36], [11.31 ± 0.18], [9732.6],
    )
  ),
  caption: [Comparação entre quantizações Q4\_K\_M e Q8\_0 para o modelo 
             Meta-Llama-3.1-8B a 32 threads.],
) <tab:quantization>
