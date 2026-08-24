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
  timing:      dwell 1757.81 ns/px   sampling 568.9 kHz   Nyquist 284.4 kHz

  [f] signal frame        [r] reference frame
  [l] line rate           [n] resolution
  [y] flyback fraction    [w] flyback in the frame: NO
  [g] run the report      [o] save last report
  [c] clear reference     [b] back to the main menu
```

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
| `ui/selection.py` | janela de seleção das áreas (bloqueante, retângulo/elipse). |
| `ui/interference_screen.py` | segunda tela: metadados de aquisição e execução do laudo. |
| `interference/` | caracterização de padrão fixo: quadro único, laudo de texto, sem gráfico. |
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
| `snr` | `snr`, `signal_mean`, `noise_mean`, `noise_std` | sim (`signal`, `noise`) |

`snr = (signal_mean − noise_mean) / noise_std`, tudo do mesmo arquivo, em
contagens brutas; a razão é adimensional. Área de ruído com variância zero vira
falha daquele arquivo, não `inf`.

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

264 testes cobrindo parsing de nomes, duplicatas, política multi-página,
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
