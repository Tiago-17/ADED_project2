#figure(
  text(size: 9pt,
    table(
      columns: (auto, auto, auto, auto, auto),
      align: (left, center, center, center, center),

      table.header(
        [*Modelo*],
        [*Quantização*],
        [*Tamanho (GB)*],
        [*$B_"efetiva"$ (GB/s)*],
        [*TPOT previsto*],
      ),
      [Llama-3.1-8B-Instruct], [Q4\_K\_M], [4.92], [18.0], [273.4],
      [TinyLlama-1.1B], [Q8\_0], [0.67], [18.0], [37.1],
      [Qwen2.5-0.5B], [Q5\_K\_M], [0.49], [18.0], [27.3],
    )
  ),
  caption: [Comparação entre modelos, quantização, tamanho e TPOT previsto.],
) <tab:model-summary>
