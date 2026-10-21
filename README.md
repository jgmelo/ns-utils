# TIF Series Analyzer

Aplicação de terminal que varre um diretório de imagens `.tif`, extrai um
parâmetro numérico do nome de cada arquivo, calcula uma métrica por imagem e
plota métrica × parâmetro.

Implementação da especificação `SPEC.md` (TIF Series Analyzer — Implementation
Specification), Python 3.10+.

```
medida_001_1mW.tif ─┐
medida_002_2mW.tif ─┼─► métrica por imagem ─► gráfico métrica × potência (mW)
medida_017_12.5mW.tiff ─┘
```

Duas métricas na v1: desvio-padrão da imagem inteira (`std`) e razão sinal-ruído
a partir de duas áreas selecionadas na tela (`snr`).

Traz também uma ferramenta separada, `tif-interference`, que caracteriza
interferência de padrão fixo num quadro e diz se a subtração entre canais é
viável — veja [Interferência de padrão fixo](#interferência-de-padrão-fixo).

## Instalação

```bash
pip install -e .          # numpy, tifffile, matplotlib
pip install -e ".[dev]"   # + readchar e pytest
```

`readchar` é dependência opcional: sem ela (ou quando o console não aceita modo
raw) a interface degrada para tecla + Enter em vez de quebrar.

## Uso

```bash
tif-analyzer                      # usa o último diretório lembrado, ou o CWD
tif-analyzer /caminho/dos/dados   # ou informe o diretório
python -m tif_analyzer.app --help # opções de linha de comando
```

Tela principal:

```
  TIF Series Analyzer

  dir:      /home/user/data/2026-08-12   (47 files, 44 matched)  3 skipped
  pattern:  medida_\d+_(?P<power>[\d.]+)mW\.tiff?$
  glob:     *.tif
  metric:   std — Standard deviation
  results:  44 points   (2 failures)
  plot:     TkAgg (interactive window)

  [d] directory        [p] pattern
  [m] metric           [g] glob filter
  [r] run analysis     [v] view plot
  [e] export CSV       [!] allow duplicates: OFF
  [a] select areas     [i] interference screen
  [f] fit a power law  [1] single file
  [s] select series
  [q] quit
```

Opções indisponíveis continuam visíveis com o motivo
(`[r] run analysis  (no matching files)`) — uma linha que some confunde.
Todo sub-prompt sai com `Esc` ou `b` sem aplicar mudanças; prompts de caminho
têm completar com Tab e histórico.

Flags úteis:

| Flag | Efeito |
|------|--------|
| `--pattern`, `--glob`, `--metric` | sobrescrevem o que estiver salvo na config |
| `--metric snr` | usa a métrica de áreas (exige seleção prévia com `[a]`) |
| `--page-policy {error,first,mean}` | política para TIFF multi-página (padrão `error`) |
| `--allow-duplicates` | não trata valores repetidos como erro |
| `--no-config` | ignora e não grava a configuração persistida |
| `--line-input` | força tecla + Enter (consoles de IDE) |
| `--no-clear` | não limpa a tela entre telas |

### Padrão de nome de arquivo

Regex com **exatamente um grupo nomeado**: o nome do grupo vira o rótulo do
eixo x e o texto capturado é convertido para float. O índice sequencial é
casado com `\d+` puro, sem nome — só o parâmetro é nomeado, senão a aplicação
não sabe qual grupo é o eixo x.

```
medida_\d+_(?P<power>[\d.]+)mW\.tiff?$
```

O prompt de padrão traz uma tabela de exemplos para editar, e valida na hora:
não compila → erro; zero grupos nomeados → "não há parâmetro para plotar";
dois ou mais → "ambíguo, qual é o eixo x?".

### Arquivo único (`[1]`)

O contrato da série — regex com um grupo nomeado cuja captura vira float — está
certo para um gráfico e errado para abrir **um** quadro e olhar. Um arquivo
chamado `poliest.tif` não tem número no nome e, até aqui, era inabrível: as
ferramentas de quadro único leem sua lista da descoberta, que só sabia produzir
registros com parâmetro.

`[1]` nomeia um arquivo direto. Sem glob, sem regex, sem checagem de duplicata —
você nomeou o arquivo, não há o que casar nem com o que colidir. O cabeçalho
troca duas linhas por uma:

```
  dir:      /dados/ensaio          (1 file)
  file:     poliest.tif
```

`pattern:` e `glob:` somem da tela porque não estão governando nada, mas
continuam no estado: sair do modo devolve a série exatamente como estava.

O registro produzido tem `param_value is None`, e o tipo é opcional de
propósito — é o que impede um arquivo sem parâmetro de chegar ao pipeline da
série disfarçado de zero plausível. A corrida continua exigindo o parâmetro;
ali ele é legítimo, é o eixo x:

```
[r] run analysis  (single file mode — no parameter to plot against)
```

O mesmo motivo aparece em `[p]`, `[g]` e `[!]`, e as quatro teclas **também não
agem** quando pressionadas — rótulo bloqueado sem guarda no dispatch deixaria
`[r]` ir até a plotagem antes de alguém notar a falta do eixo.

`[a]`, `[i]`, `[s]` e `[f]` passam a funcionar normalmente. O rótulo da tecla
nomeia a ação, não o estado: `[1] single file` quando fora, `[1] back to the
series` quando dentro. Não persiste na config — é modo de inspeção, como o corte
de beads em `[k]`.

### Seleção de áreas (sinal e ruído)

A métrica `snr` mede duas regiões da mesma imagem. Você seleciona as duas **uma
vez**, numa imagem de referência que sabe conter o sinal, e o cálculo roda sobre
todos os arquivos válidos:

```
[a] ─► escolhe a imagem de referência ─► janela abre
       │  arrasta a área de SINAL  → [enter]
       │  arrasta a área de RUÍDO  → [enter]
       ▼
   .tif-analyzer-rois.json gravado no diretório dos dados
```

Teclas dentro da janela:

| Tecla | Efeito |
|-------|--------|
| arrastar | desenha a área da etapa atual |
| `r` / `e` | retângulo ou elipse |
| `enter` | confirma a área e passa para a próxima |
| `u` | desfaz (a caixa pendente, depois a confirmada) |
| `l` | alterna o realce de exibição (gamma ↔ linear) |
| `esc` | cancela tudo, nada é alterado |

A imagem aparece com clipe de percentil 1–99% e realce gamma, senão um quadro de
16 bits com um feixe compacto vira preto puro e não dá para julgar onde está o
fundo. **É curva de exibição apenas** — a métrica sempre vê contagens brutas.

As áreas ficam num sidecar `.tif-analyzer-rois.json` ao lado das imagens, não na
config global: são coordenadas em pixels, só fazem sentido para aquele conjunto.
Ao entrar num diretório elas são carregadas e conferidas contra a dimensão real
das imagens; se não baterem, um aviso explica e elas **não** são aplicadas.

Imagem com dimensão diferente da referência vira falha daquele arquivo, coletada
como qualquer outra — a corrida não aborta.

Seleção exige interface gráfica. Sem ela, `[a]` fica desabilitada com o motivo;
áreas já gravadas no sidecar continuam funcionando, então dá para rodar e
exportar por SSH, só não selecionar.

### Ajuste de lei de potência (`[f]`)

Responde: *esta série está limitada por shot noise ou pelo detector?* A física
dá o modelo — `y = a·xᵇ`, e o expoente é a resposta:

| relação | b se shot | b se detector |
|---|---|---|
| variância × média, dentro do ROI de sinal | 1,0 | 0,0 |
| SNR × potência | 0,5 | 1,0 |

O ajuste é mínimos quadrados em log-log (a lei de potência vira reta), com `b`
livre **e** os dois modelos de expoente travado para comparação de resíduo.

**O painel que decide não é o do SNR.** O `snr` divide por `noise_std`, medido
na área de ruído — uma área escolhida justamente por não ter a amostra. Aquele
sigma não cresce quando a amostra fica mais brilhante, então
`SNR = S/σ_fundo ∝ P` sai linear por construção. Medido em quadros Poisson
sintéticos: σ da área de ruído parado em 31 contagens enquanto a potência variava
16×, e σ dentro da área de sinal indo de 36 a 75.

Quem decide é a **curva de transferência de fótons**: variância × média, ambas
dentro do ROI de sinal. Poisson diz que *variância = média*, então shot noise dá
expoente 1 e ruído de leitura dá 0. E não é o mesmo que σ × sinal-acima-do-fundo:
a área de sinal contém também os fótons do fundo, então `σ = √(fundo + sinal)`
não é lei de potência no sinal — nos mesmos quadros Poisson essa forma devolveu
0,27 e veredito *neither*, enquanto a curva de transferência devolveu 0,99.

Três ajustes, e um terceiro veredito de contexto:

```
  photon transfer curve: signal variance vs signal mean
    b = +0.992 ± 0.008   (3·SE: [+0.97, +1.02])
    5 point(s), lever arm 0.65 decades
      b = 0 locked: residual rms 0.2316
      b = 1 locked: residual rms 0.0039
    → shot-noise limited: b ± 3·SE = [0.97, 1.02] contains 1 and excludes ...

  VERDICT: shot-noise limited
    · the b = 1 model's amplitude is the system gain: 0.986 counts per electron
```

**Os quatro guardas**, porque com 4 ou 5 pontos um expoente de 0,62 pode ser
compatível com 0,5 *e* com 1,0 ao mesmo tempo:

- **Erro-padrão do expoente**, `σ_resíduo / √Σ(log xᵢ − log x̄)²`. O denominador é
  a alavanca — não o espalhamento, que é o numerador. Afastar os pontos em x
  encolhe o erro do expoente mesmo com dado ruidoso.
- **A alavanca em décadas.** Potências dentro de um fator de 3 não separam 0,5 de
  1,0 por mais limpa que esteja a medida, e o laudo diz isso em vez de escolher.
  Vale para o eixo da curva de transferência também: com fundo de 1000 contagens
  e sinal de algumas centenas, a média mal se move e não há o que ajustar.
- **Veredito só com um candidato dentro do intervalo.** Conter os dois é uma
  resposta de verdade — *indeterminado* —, e conter nenhum também.
- **Pontos sem logaritmo** (SNR ≤ 0) são descartados **e contados**.

Duas ressalvas viajam com a curva de transferência: o espalhamento dentro da
área de sinal só é ruído se aquela área for uniforme; e a média é em contagens
cruas, então um offset eletrônico empurra o expoente para baixo de 1 — separar
isso exigiria um quadro escuro.

**`[x]` corta o topo da série.** Acima de certa potência a resposta satura ou a
amostra branqueia, e aqueles pontos arrastam o expoente. `[x]` define a potência
máxima a incluir; o corte tira o **arquivo inteiro** dos três ajustes, porque a
mesma medida alimenta os três painéis e uma série em que a curva de transferência
viu um quadro que o ajuste de SNR não viu seriam duas análises apresentadas como
uma.

Os pontos cortados continuam **desenhados**, vazados, com a curva ajustada
estendida por cima deles — o motivo de cortar costuma ser que o topo foge da
reta, e esconder a fuga esconde o motivo. Num exemplo com joelho em 40 mW, o
ajuste da resposta vai de `P^0,63` (indeterminado) para `P^1,00` ao cortar.

O corte tem preço e ele aparece: menos pontos e menos alavanca. No mesmo
exemplo, a curva de transferência cai para 0,42 década e passa a dizer
*indeterminate* — corretamente.

`[l]` alterna log-log, `[v]` desenha os três painéis, `[o]` grava o laudo. A tela
recalcula do relatório a cada desenho: não existe um quarto estado derivado para
ficar desatualizado. Como o corte de beads em `[k]`, o corte de potência é knob
de análise da sessão — não vai para sidecar nenhum.

### Dados de exemplo

```bash
python scripts/make_sample_series.py /tmp/sample --with-unmatched --with-stack
tif-analyzer /tmp/sample

# com um feixe gaussiano, para exercitar a métrica de áreas
python scripts/make_sample_series.py /tmp/beam --beam --size 160
tif-analyzer /tmp/beam --metric snr
```

## Interferência de padrão fixo

Ferramenta de **quadro único**, separada do fluxo de séries: um TIFF entra, um
laudo de texto sai. Responde a uma pergunta só — *dá para cancelar essa
interferência subtraindo o canal de referência?*

```bash
tif-interference sinal.tif -r referencia.tif --line-rate 1000 --flyback 0.1
```

Sem `--line-rate` tudo funciona, menos a conversão para Hz, que é declarada como
não calculada em vez de chutada.

No TUI, `[i]` abre uma **tela própria** onde se preenche tudo antes de rodar:

```
  Interference — fixed-pattern characterisation

  signal:      medida_002_30mW[PMT2].tif
  reference:   medida_003_30mW[PMT1].tif
  line rate:   1000 Hz
  resolution:  (frame width: 512 px)
  flyback:     10.0%   (in the frame: no)
  bead cut:    median + 3σ   (default)
  timing:      dwell 1757.81 ns/px   sampling 568.9 kHz   Nyquist 284.4 kHz

  [f] signal frame        [r] reference frame
  [l] line rate           [n] resolution
  [y] flyback fraction    [w] flyback in the frame: NO
  [k] bead cut (k sigmas)
  [g] run the report      [v] view the bead mask
  [p] view profile & spectrum   [s] subtract the pattern (one frame or the series)
  [o] save last report    [c] clear reference
  [b] back to the main menu
```

`[k]` ajusta o corte da máscara de beads, em sigmas robustos acima da mediana
do fundo: mais alto mantém mais quadro como fundo, mais baixo corta com mais
agressividade. É um parâmetro de análise, não de aquisição, então **não** vai
para o sidecar — a ideia é ajustar olhando a figura e, quando o valor assentar,
fixá-lo com `--mad-k` na CLI.

`[v]` abre a figura da máscara de beads: o quadro com os pixels rejeitados
pintados por cima, a fração rejeitada por coluna alinhada ao eixo x da imagem
(um pico fica embaixo da conta que o causou), e o histograma de valores com a
mediana e o corte. O laudo diz *quanto* foi rejeitado; a figura diz *onde*, e se
o corte está num lugar razoável. Janela bloqueante, com o mesmo fallback do
gráfico de séries: sem GUI, salva um PNG e informa o caminho.

`[p]` abre o perfil ao lado do espectro: à esquerda a mediana por coluna que a
FFT leu, com a versão crua atrás (a diferença entre as duas é a contaminação por
beads que o laudo quantifica); à direita o espectro de potência com o cluster
sombreado, o limiar de detecção desenhado e um eixo de período no topo; embaixo,
alguns períodos com a componente dominante sobreposta, ajustada por projeção na
frequência do centroide. O desvio residual dessa sobreposição diz se **uma**
componente descreve o perfil ou se falta coisa.

A linha `timing:` recalcula a cada mudança, **antes** de rodar qualquer coisa:
um line rate errado aparece como um Nyquist errado, não como uma resposta
errada. Os campos numéricos aceitam `none` para limpar, e o flyback aceita
tanto `0.1` quanto `10%`.

Esses metadados ficam num sidecar `.tif-analyzer-acquisition.json` no diretório
dos dados, pelo mesmo motivo das áreas: descrevem como aquele conjunto foi
adquirido, não como você gosta de trabalhar. Redigitar o line rate toda sessão
é o jeito mais rápido de acabar com um erro de digitação numa frequência
publicada.

Código de saída: `0` viável, `3` não viável, `4` inconclusivo, `1`/`2` erro.

### As sete operações

| # | O que faz |
|---|-----------|
| 1 | Carrega em float64 sem reescalar; registra shape, dtype de disco, min/mediana/max e satura&shy;ção no limite do tipo |
| 2 | Máscara de beads por mediana/MAD iterativo; fração rejeitada global **e** da pior coluna |
| 3 | Perfil de padrão fixo = mediana por coluna, centrada; variante mascarada e a discrepância entre as duas |
| 4 | FFT janelada, cluster dominante, centroide ponderado por potência, período, largura, fração da potência |
| 5 | Hz a partir do dwell, **família de alias** e limite de Nyquist declarado |
| 6 | Fase por linha: deriva em °/linha e estabilidade de amplitude |
| 7 | Comparação entre canais: período, razão de amplitude, Pearson bruto **e** restrito à banda, offset de fase por Hilbert |

### Convenção de eixos

Cada **linha** é uma linha de varredura e o índice de coluna é tempo, então a
interferência aparece como listras verticais e o perfil é a mediana por coluna.
O laudo mede a força da listra nos dois eixos e avisa se o padrão for mais forte
no outro — analisar o eixo errado produz números plausíveis e errados.

### Flyback

Uma flag decide duas coisas acopladas. Sem `--frame-includes-flyback`, as
colunas salvas cobrem só a parte ativa: `t_dwell = T_linha·(1−flyback)/N` e
nenhuma coluna é descartada. Com ela, as colunas cobrem o período inteiro:
`t_dwell = T_linha/N` e as colunas de retorno saem do perfil.

### O que o veredito diz, e o que ele não diz

Viável quando os períodos concordam dentro da tolerância **e** a correlação
restrita à banda passa do limiar. Cada verificação aparece com **o valor ao lado
do limiar**, para você discordar sem ler código.

Dois avisos independentes do veredito:

- **Lacuna entre correlação bruta e em banda.** Se as listras batem mas o resto
  não, subtrair o quadro cru cancelaria o padrão e injetaria o ruído de banda
  larga da referência no lugar.
- **Correlação em banda negativa.** A referência carrega o padrão invertido:
  precisa ser somada, não subtraída.

A deriva de fase separa dois casos com consequências diferentes: **≈ 0** e o
padrão está travado no gatilho de linha, então repete entre quadros e um padrão
guardado serviria; **≠ 0** e ele anda entre quadros, então só referência
simultânea cancela.

**Todos os limiares são provisórios** — escolhidos por princípio, não calibrados
contra quadros reais — e o laudo diz isso em toda execução. Ficam em
`InterferenceParams`, ajustáveis por flag.

### Alias

A família inteira é emitida com o Nyquist explícito, porque num scan de 1 kHz
60 Hz e 940 Hz são indistinguíveis a partir das amostras. Cada candidato traz a
deriva de fase por linha que produziria, o que às vezes escolhe um deles — mas a
previsão é sensível ao centroide, e quando a incerteza passa de meia volta o
laudo declara que **não dá para separar** em vez de apontar um vencedor.

### Subtração do padrão (`[s]`)

Terceira tela, alcançada de `[i]`. Assume que a subtração é viável — não
re-discute o veredito — mas sempre mostra o que a correção fez antes de você
guardar o arquivo. **Nada é escrito até `[w]`.**

```
  mode:        profile — signal minus its own column median (no reference)

  computed:
    period 16.000 px
    stripe      39.35 →   0.7129 counts   (98.2% removed)
    noise       2.965 →    2.965 counts   (+0.0%)
```

Dois modos:

| modo | conta | quando |
|---|---|---|
| `reference` | `S − α·R_alinhado` | referência simultânea disponível |
| `profile` | `S − perfil(x)` | padrão travado na linha; não gasta canal |

No modo `reference`: a referência é alinhada por deslocamento sub-pixel de
Fourier, escalada por **mínimos quadrados sobre os pixels de fundo** (beads
excluídos, senão um grão brilhante puxa α e a subtração come sinal real), e por
padrão **restrita à banda da listra** — subtrair a referência crua cancela a
listra e injeta o ruído de banda larga dela no lugar. `[e]` ajusta α linha a
linha em vez de uma vez para o quadro.

As duas linhas de `computed:` são o que decide se vale a pena: quanto de listra
saiu **e** quanto de ruído entrou. Uma correção que tira 90% da listra e dobra o
ruído não é ganho. Se a listra *aumentar*, um aviso alto aparece — não é veredito
de viabilidade, é conferência de aritmética.

Números típicos de um quadro sintético com listra de 40 contagens e referência
ruidosa, para calibrar expectativa:

| configuração | listra removida | ruído |
|---|---|---|
| perfil | 98% | +0% |
| referência, em banda | 89% | +84% |
| referência, crua | 32% | +360% |

A saída é **float32**: os valores corrigidos são fracionários (α é), e uint16
arredondaria cada pixel, jogando fora parte da distribuição de ruído que a
medida de SNR vai justamente medir. O nível de fundo original é preservado. Os
parâmetros vão para a descrição do TIFF **e** um sidecar
`<nome>.subtraction.json` — imagem corrigida sem registro de como é armadilha.

Se a imagem corrigida abrir **toda branca**, é a escala de exibição: float32
com fundo em ~1000 contagens num visualizador que espera 0–1. No ImageJ,
`Image > Adjust > Brightness/Contrast > Auto`.

#### A série inteira (`[a]`)

`[a]` aplica o mesmo plano — o que está na tela, com os mesmos toggles — a
**todos os arquivos que casam com o glob e o regex atuais**, para a série
corrigida poder ser medida e plotada como qualquer outra.

```
  [1/3] ✓ medida_001_20mW.tif:  55.2% of the stripe removed, noise +0.0%
  [2/3] ✓ medida_002_40mW.tif:  57.1% of the stripe removed, noise -0.1%
  [3/3] ✓ medida_003_60mW.tif:  54.8% of the stripe removed, noise +0.0%

  ✓ 3 frame(s) written to /dados/deprofiled
    summary table: subtraction-summary.csv
    stripe removed: 54.8% … 57.1% across the series
    point [d] at that directory and run the series as usual.
```

Duas decisões que essa tela toma por você:

- **Diretório próprio** (`deprofiled/` ou `corrected/`), nunca ao lado dos
  originais — dois arquivos com o mesmo valor de parâmetro na mesma pasta é
  exatamente o que a descoberta recusa, e a série corrigida nasceria
  improcessável.
- **Nomes preservados**, para o glob e o regex que acharam os originais acharem
  os corrigidos sem você mexer em nada. Os sidecars de áreas e de aquisição
  vão junto, porque as áreas são coordenadas em pixel e os quadros mantiveram
  o tamanho: `[d]` na pasta nova e `[r]` já roda.

Além do `.subtraction.json` de cada quadro, a corrida grava um
`subtraction-summary.csv` com uma linha por arquivo. É o que responde a
pergunta que um gráfico de série depende: **os quadros foram corrigidos de
maneira parecida?** Uma corrida que vai de 20% a 90% de remoção colocou na
série uma tendência que a amostra não tem — a tela avisa quando a faixa passa
de 15 pontos percentuais.

No modo `reference`, o *mesmo* quadro de referência serve para todos os
arquivos (e é pulado se estiver na lista). Só é defensável com padrão travado;
com deriva entre aquisições, cada sinal precisaria da sua própria referência
simultânea.

### Dados de exemplo

```bash
python scripts/make_interference_frames.py /tmp/if --drift 0.3 --beads
tif-interference /tmp/if/signal.tif -r /tmp/if/reference.tif --line-rate 1000
```

## Uso como biblioteca

Nada fora de `tif_analyzer/ui/` lê ou escreve no terminal — a camada de análise
roda em script ou notebook sem UI nenhuma:

```python
from tif_analyzer import analysis, discovery, metrics

report = discovery.discover("/dados/2026-08-12")
run = analysis.run_blocking(report.records, metrics.get("std"))

for result in run.results:
    print(result.record.param_value, result.values["std"])
print(run.failures)
```

Para acompanhar progresso, `analysis.run` é um gerador: ele emite um
`ProgressEvent` por arquivo e devolve o `RunReport` como valor de retorno
(`StopIteration.value`).

## Arquitetura

| Módulo | Responsabilidade |
|--------|------------------|
| `discovery.py` | glob + regex + `FileRecord`s + duplicatas. Nenhum I/O de imagem. |
| `regions.py` | `Rect`/`Ellipse`/`RegionSet`, máscara, extração de pixels, hash das áreas. |
| `roi_sidecar.py` | leitura/escrita de `.tif-analyzer-rois.json` no diretório dos dados. |
| `acquisition_sidecar.py` | leitura/escrita de `.tif-analyzer-acquisition.json` (line rate, flyback). |
| `loading.py` | abre TIFFs via `tifffile`, aplica a política multi-página, devolve `float64`. |
| `preprocess.py` | pipeline ordenado + serialização canônica + hash. Vazio na v1. |
| `metrics.py` | registro de métricas via `@register`. Nada fora daqui lista métricas. |
| `analysis.py` | orquestra load → preprocess → compute. Gerador; coleta falhas. |
| `cache.py` | cache em memória, chave `(path, mtime_ns, size, metric_key, preprocess_hash)`. |
| `plotting.py` | figura matplotlib + detecção de backend headless. |
| `export.py` | CSV formato longo com o snapshot em linhas `#`. |
| `config.py` | configuração persistida (último diretório, padrão, métrica, glob, duplicatas). |
| `state.py` | `AppState` + regra de invalidação. |
| `fitting.py` | lei de potência em log-log, erro-padrão do expoente, veredito com recusa. |
| `noise_regime.py` | a física: shot × detector, a curva de transferência e as conferências. |
| `ui/selection.py` | janela de seleção das áreas (bloqueante, retângulo/elipse). |
| `ui/fit_screen.py` | tela do ajuste: expoente, veredito de regime, três painéis. |
| `ui/interference_screen.py` | segunda tela: metadados de aquisição e execução do laudo. |
| `ui/subtraction_screen.py` | terceira tela: calcula, mostra, grava a subtração e aplica à série. |
| `interference/` | caracterização de padrão fixo: quadro único, laudo de texto. |
| `interference/figures.py` | figuras da caracterização (máscara, perfil + espectro, antes/depois). |
| `interference/subtract.py` | alinhamento, escala, subtração e as medidas de antes/depois. |
| `interference/writing.py` | grava o quadro corrigido em float32 com procedência. |
| `interference/batch.py` | corrige um quadro e corrige a série inteira, pela mesma função. |
| `ui/` | loop de menu, teclas, prompts, renderização. Único lugar com stdin/stdout. |
| `app.py` | entry point. |

### Decisões travadas da SPEC e onde estão

| # | Decisão | Onde |
|---|---------|------|
| 1 | Multi-página → um ponto por arquivo; política `first`/`mean`/`error`, padrão `error` | `loading.load_image` |
| 2 | Métricas devolvem `dict[str, float]`, nunca float puro | `metrics.compute_validated` |
| 3 | Preprocessing é lista ordenada de passos, vazia na v1 | `preprocess.apply_pipeline` |
| 4 | Duplicatas são erro duro, na descoberta, antes de qualquer I/O | `discovery.discover` |
| 5 | Resultados em cache por `(path, mtime_ns, size, metric_key, preprocess_hash, regions_hash)` | `cache.make_key` |
| 6 | Plot interativo, com fallback para PNG em backend não-GUI | `plotting.show_or_save` |
| 7 | Último diretório persistido e oferecido como padrão | `config.resolve_startup` |

### Regra de invalidação

Alterar `directory`, `glob`, `pattern`, `metric_key`, `preprocess_steps`,
`allow_duplicates`, `page_policy` ou `regions` limpa `records`, `discovery` e
`report` — redesenhar uma área invalida os números anteriores.
Alterar `selected_series` **não** limpa nada — só afeta a renderização. Toda
mutação passa por `AppState.update`, uma função só, porque bug de gráfico
desatualizado nasce exatamente de atribuições espalhadas.

### Tratamento de erro

- Falha por arquivo não aborta a execução: vai para `RunReport.failures` e é
  apresentada em bloco no fim, com os resultados parciais utilizáveis.
- Problemas de descoberta (nenhum match, duplicatas) bloqueiam a execução e
  explicam o motivo.
- Erro de programação (métrica devolvendo chaves fora do que declarou) sobe como
  `MetricContractError` — não se defende de bug próprio engolindo o erro.

### Exportação CSV

Formato longo, com o snapshot de configuração em linhas comentadas:

```
# directory: /dados/2026-08-12
# metric: std
# pattern: medida_\d+_(?P<power>[\d.]+)mW\.tiff?$
filename,param_name,param_value,output_key,value
medida_001_1mW.tif,power,1,std,12.4471
```

## Métricas

| key | saídas | precisa de áreas |
|-----|--------|------------------|
| `std` | `std` (desvio amostral, ddof=1, imagem inteira) | não |
| `snr` | `snr`, `signal_mean`, `noise_mean`, `noise_std`, `signal_std` | sim (`signal`, `noise`) |

`snr = (signal_mean − noise_mean) / noise_std`, tudo do mesmo arquivo, em
contagens brutas; a razão é adimensional. Área de ruído com variância zero vira
falha daquele arquivo, não `inf`.

`signal_std` é o desvio **dentro** da área de sinal, que a razão não usa. Está
ali porque é a única saída que carrega a estatística de fótons da própria
amostra — a área de ruído não tem fóton nenhum dela — e é o que o ajuste `[f]`
usa para decidir o regime de ruído.

### Adicionando uma métrica

Só `metrics.py` é tocado:

```python
@register(
    key="contrast",
    name="Contrast",
    description="Sinal menos fundo, em contagens brutas.",
    outputs=(OutputSpec("contrast", "Contrast", "counts"),),
    requires_regions=("signal", "noise"),      # opcional; omita para métrica global
)
def _contrast(image, regions):                  # sem regiões, a assinatura é (image)
    return {"contrast": float(regions["signal"].mean() - regions["noise"].mean())}
```

`outputs` e `requires_regions` são declarados estaticamente, então a UI lista as
séries em `[s]` e sabe dizer "selecione as áreas primeiro" sem rodar nada antes.
As regiões chegam como amostras 1-D já extraídas — média e desvio não se importam
com forma, e elipse não tem sub-imagem retangular. Uma métrica que precise da
geometria 2-D da região precisaria de mais que isso.

## Testes

```bash
python -m pytest
```

319 testes cobrindo parsing de nomes, duplicatas, política multi-página,
contrato das métricas, geometria e máscaras das áreas, hash e invalidação de
cache, sidecar corrompido ou com dimensão errada, máquina de estados da seleção,
config corrompida, CSV, fallback headless do gráfico e o loop de menu com teclas
roteirizadas. A parte de interferência é testada contra verdade plantada:
quadros sintéticos com período, fase, deriva e offset conhecidos, mais os casos
que têm de falhar — ruído puro sem detecção, eixo trocado, shapes divergentes,
alias indecidível.

## Fora de escopo na v1

Passos de preprocessing com comportamento real (o pipeline existe, vazio), cache
persistente em disco, extração de dois parâmetros / séries agrupadas, varredura
recursiva de diretórios, formatos não-TIFF, mais de duas áreas, áreas diferentes
por arquivo, polígono livre e subtração de dark-frame.
