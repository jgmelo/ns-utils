# Resumo — ns-utils

Duas ferramentas, um pacote Python 3.10+, 456 testes.

## 1. TIF Series Analyzer (`tif-analyzer`)

Varre um diretório, extrai um parâmetro do **nome do arquivo** por regex com
exatamente um grupo nomeado, calcula uma métrica por imagem, plota
métrica × parâmetro.

| métrica | saídas | áreas |
|---|---|---|
| `std` | desvio amostral (ddof=1) da imagem inteira | não |
| `snr` | `snr`, `signal_mean`, `noise_mean`, `noise_std`, `signal_std` | sim |
| `snr_signal` | mesma razão, dividida por `signal_std` | sim |

`snr = (signal_mean − noise_mean) / noise_std`, em contagens brutas.
`signal_std` não entra na razão de `snr`: é o desvio *dentro* da área de sinal,
única saída que carrega a estatística de fótons da amostra. `snr_signal` divide
por ele — exige área uniforme, e seus expoentes 0,5/1,0 são assíntotas: com
pedestal de fundo o valor fica entre os dois (0,73 medido sobre 1000 contagens).

`[a]` seleciona duas áreas (retângulo/elipse) numa imagem de referência,
aplicadas a toda a série; ficam em `.tif-analyzer-rois.json` no diretório dos
dados e são conferidas contra a dimensão real das imagens.

**Decisões estruturais**

- TIFF multi-página nunca vira um número em silêncio: `error` / `first` / `mean`.
- Métricas devolvem `dict[str, float]`, nunca float puro.
- Duplicatas de parâmetro são erro duro na descoberta, antes de qualquer I/O,
  agrupadas por float parseado (`5mW` = `5.00mW` = `05mW`).
- Cache por `(path, mtime_ns, size, metric_key, preprocess_hash, regions_hash)`:
  mover uma área invalida resultados; redesenhá-la noutro arquivo, não.
- Toda mudança passa por `AppState.update` e limpa resultados.
- Três contagens sempre reportadas: glob → regex → float.
- Falha por arquivo é coletada, não aborta a corrida.
- **`[1]` arquivo único**: nomeia um quadro direto, sem glob, regex nem
  duplicata. `param_value` é `None` — o tipo opcional é que impede um arquivo
  sem parâmetro de entrar na série disfarçado de zero. `[r]` vira *measure this
  frame*: mesma corrida e mesma cache, laudo com todas as saídas da métrica em
  vez de gráfico. Só o que precisa de eixo x sai de cena — `[v]`, `[e]`, `[f]`,
  `[p]`, `[g]`, `[!]` — bloqueados **e inertes**. O `pattern` e o `glob`
  sobrevivem ao modo, só saem da tela.

### Ajuste (`[f]`)

`y = a·xᵇ` por mínimos quadrados em log-log, com `b` livre e os dois modelos
travados. Decide **shot × detector**.

- **Quem decide é a curva de transferência de fótons** (variância × média, dentro
  da área de sinal): Poisson dá variância = média, logo b = 1 shot, b = 0
  detector. O SNR **não** decide: divide por σ da área de ruído, que não tem
  fóton da amostra, então `SNR ∝ P` sai linear por construção. Verificado em
  quadros Poisson: σ do ruído parado em 31 enquanto σ do sinal ia de 36 a 75.
- **Nem σ × sinal-acima-do-fundo**: a área de sinal contém o fundo, então
  `σ = √(fundo + sinal)` não é lei de potência. Nos mesmos quadros deu 0,27 e
  *neither*; a curva de transferência deu 0,99.
- **Erro-padrão do expoente** = `σ_resíduo / √Σ(log xᵢ − log x̄)²`. Denominador é
  a alavanca, numerador é o espalhamento. Afastar os x encolhe o erro de `b`.
- **Recusa explícita**: intervalo que contém os dois candidatos = *indeterminado*.
  Alavanca menor que ~0,45 década = veredito retido, por mais limpo que seja o
  ajuste. Fundo alto come a alavanca da curva de transferência.
- Onde a curva é limpa, a amplitude do modelo b = 1 é o **ganho** em
  contagens/elétron.
- **`[x]` corta o topo da série** (saturação, bleaching). Tira o arquivo inteiro
  dos três ajustes — corte por painel seriam duas análises apresentadas como uma
  — e os pontos cortados continuam desenhados, vazados, com a curva estendida
  por cima. O preço aparece: menos alavanca, e o veredito pode virar
  *indeterminado*.

## 2. Interferência de padrão fixo (`tif-interference`, tecla `[i]`)

Quadro único + referência opcional → laudo de texto. Responde: *dá para cancelar
a listra subtraindo o outro canal?* Não corrige nada.

**Eixos**: cada linha é uma varredura, coluna é tempo; o perfil é a **mediana
por coluna**. A força da listra é medida nos dois eixos, com aviso.

**Pipeline**: carga em float64 sem reescalar → máscara de beads → perfil →
espectro → Hz e alias → fase por linha → comparação entre canais → veredito.

**Conceitos que decidem a leitura**

- **Máscara**: mediana/MAD iterativo, `sigma = 1,4826 × MAD`, corte unilateral em
  `mediana + k·sigma` (`[k]`). Reporta rejeição global **e** da pior coluna.
- **Cluster**: grupo de bins vizinhos ao pico — uma frequência real nunca cai
  exatamente num bin. Centroide ponderado por potência → período.
- **Limiar de detecção**: num perfil de ruído a potência de cada bin é
  *exponencial* (média plana, cauda longa), e o maior de N bins fica ~9× a
  mediana por acaso. O limiar vem de `log₂(N/α)`, α = 1% → ~15×.
- **Alias**: com Nyquist explícito, porque num scan de 1 kHz 60 Hz e 940 Hz são
  indistinguíveis das amostras.
- **Deriva ≈ 0** → padrão travado no gatilho de linha, repete entre quadros, um
  padrão guardado serviria. **Deriva ≠ 0** → só referência simultânea cancela.
  Corolário: se o padrão sobrevive à mediana sobre todas as linhas, está travado.
- **Correlação bruta ≪ em banda**: listras batem, o resto não → subtrair cru
  injetaria o ruído de banda larga da referência.
- **Variância explicada**: amplitude de senoide → `std = A/√2`; variâncias somam
  (Pitágoras), desvios não.
- **Pico largo ⟺ amplitude modulada** (dualidade de Fourier). Cluster com pouca
  fração da potência = uma componente não descreve o perfil.

**Limiares provisórios** (2% de período, 0,80 de correlação em banda, 0,30 de
lacuna, k = 3): escolhidos por princípio, **não calibrados**. Todo laudo diz isso.

## 3. Subtração do padrão (`[s]`)

`reference`: `S − α·R_alinhado`, com alinhamento sub-pixel por Fourier, α por
mínimos quadrados sobre pixels de fundo, e a referência restrita à banda da
listra (crua injeta o ruído de banda larga dela). `profile`: `S − perfil(x)`,
sem referência, só funciona se o padrão estiver travado na linha.

Sempre reporta **listra removida e ruído injetado** — uma correção que tira 90%
da listra e dobra o ruído não é ganho. Saída float32 com procedência no TIFF e
num sidecar; nível de fundo original preservado.

`[a]` aplica o mesmo plano a todos os arquivos que casam com glob + regex, pela
**mesma função** que a tela usa num quadro só — a tela é preview do lote ou não
serve. Saída num diretório próprio (ao lado dos originais seria duplicata de
parâmetro) e com os nomes preservados (para o mesmo regex achá-los). Sidecars de
áreas e aquisição vão junto. Um `subtraction-summary.csv` responde a pergunta
que a série depende: os quadros foram corrigidos de maneira parecida? Faixa de
remoção larga = tendência artificial na série.

Sinal do deslocamento: a referência anda o **oposto** do offset medido. Errar o
sinal deixa a listra intacta e produz um α plausível — por isso vive numa função
só, `alignment_shift`, com teste.

## 4. Figuras

- `[v]` **máscara**: quadro com rejeitados pintados, fração por coluna alinhada
  ao eixo da imagem, histograma com mediana e corte.
- `[p]` **perfil + espectro**: perfil mascarado e cru, espectro com cluster e
  limiar, zoom com a componente ajustada **por projeção** — inverter os bins do
  cluster perde amplitude para vazamento.
- Exibição com clipe de percentil 1–99% + gamma 0,4. **Só exibição**: a métrica
  sempre vê contagens brutas.

## Pendências

Calibrar limiares com dado real, inclusive a regra de 3·SE do ajuste; STFT do
perfil para separar modulação de amplitude de velocidade de varredura não
uniforme; ler metadados do TIFF.
