#figure(
  text(size: 9pt,
    table(
      columns: (auto, auto, auto, auto),
      align: (center, center, center, center),

      table.header(
        [*Threads*],
        [*TPOT observado*],
        [*TPOT previsto*],
        [*Erro relativo*],
      ),
      [4],  [647.12], [255.6], [+153.2%],
      [8],  [329.12], [255.6], [+28.8%],
      [16],  [170.52], [255.6], [−33.3%],
      [32],  [90.41], [255.6], [−64.6%],
      [48],  [88.99], [255.6], [−65.2%],
    )
  ),
  caption: [Comparação entre TPOT observado e previsto para diferentes números de threads.],
) <tab:tpot-threads>
