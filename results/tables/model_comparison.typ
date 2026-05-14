#figure(
  text(size: 9pt,
    table(
      columns: (auto, auto, auto, auto, auto, auto),
      align: (left, center, center, center, center, center),

      table.header(
        [*Model*],
        [*Category*],
        [*TTFT (s)*],
        [*TPOT (ms)*],
        [*Throughput\ (tok/s)*],
        [*Peak Mem.\ (MB)*],
      ),

      // Llama-3.1-8B
      table.cell(rowspan: 3)[Meta-Llama-3.1-8B\
        (Q4\_K\_M)],
      [Short],  [0.78 ± 0.22], [90.41 ± 0.05], [11.17 ± 0.00], [5838.8],
      [Medium],  [2.73 ± 1.25], [90.79 ± 0.19], [11.12 ± 0.02], [5972.7],
      [Long],  [25.04 ± 5.64], [95.74 ± 1.22], [10.55 ± 0.14], [6072.3],

      // Qwen2.5-0.5B
      table.cell(rowspan: 3)[Qwen2.5-0.5B],
      [Short],  [0.12 ± 0.03], [17.00 ± 0.10], [59.97 ± 1.21], [661.0],
      [Medium],  [0.37 ± 0.21], [17.12 ± 0.07], [59.01 ± 0.23], [673.9],
      [Long],  [3.42 ± 0.68], [17.95 ± 0.08], [56.28 ± 0.24], [689.5],

      // TinyLlama-1.1B
      table.cell(rowspan: 3)[TinyLlama-1.1B],
      [Short],  [0.14 ± 0.04], [17.10 ± 0.06], [59.30 ± 0.57], [849.5],
      [Medium],  [0.47 ± 0.27], [17.19 ± 0.06], [59.55 ± 0.61], [858.6],
      [Long],  [5.28 ± 1.42], [19.80 ± 0.07], [52.19 ± 0.46], [880.4],
    )
  ),
  caption: [Comparação de métricas de inferência entre modelos a 32 threads.],
) <tab:model-comparison>
