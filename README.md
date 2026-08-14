# TIF Series Analyzer

Aplicação de terminal que varre um diretório de imagens `.tif`, extrai um
parâmetro numérico do nome de cada arquivo, calcula uma métrica por imagem e
plota métrica × parâmetro.

Implementação da especificação `SPEC.md` (TIF Series Analyzer — Implementation
Specification), Python 3.10+.

```
medida_001_1mW.tif ─┐
medida_002_2mW.tif ─┼─► desvio-padrão por imagem ─► gráfico std × potência (mW)
medida_017_12.5mW.tiff ─┘
```

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

### Dados de exemplo

```bash
python scripts/make_sample_series.py /tmp/sample --with-unmatched --with-stack
tif-analyzer /tmp/sample
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
| `loading.py` | abre TIFFs via `tifffile`, aplica a política multi-página, devolve `float64`. |
| `preprocess.py` | pipeline ordenado + serialização canônica + hash. Vazio na v1. |
| `metrics.py` | registro de métricas via `@register`. Nada fora daqui lista métricas. |
| `analysis.py` | orquestra load → preprocess → compute. Gerador; coleta falhas. |
| `cache.py` | cache em memória, chave `(path, mtime_ns, size, metric_key, preprocess_hash)`. |
| `plotting.py` | figura matplotlib + detecção de backend headless. |
| `export.py` | CSV formato longo com o snapshot em linhas `#`. |
| `config.py` | configuração persistida (último diretório, padrão, métrica, glob, duplicatas). |
| `state.py` | `AppState` + regra de invalidação. |
| `ui/` | loop de menu, teclas, prompts, renderização. Único lugar com stdin/stdout. |
| `app.py` | entry point. |

### Decisões travadas da SPEC e onde estão

| # | Decisão | Onde |
|---|---------|------|
| 1 | Multi-página → um ponto por arquivo; política `first`/`mean`/`error`, padrão `error` | `loading.load_image` |
| 2 | Métricas devolvem `dict[str, float]`, nunca float puro | `metrics.compute_validated` |
| 3 | Preprocessing é lista ordenada de passos, vazia na v1 | `preprocess.apply_pipeline` |
| 4 | Duplicatas são erro duro, na descoberta, antes de qualquer I/O | `discovery.discover` |
| 5 | Resultados em cache por `(path, mtime_ns, size, metric_key, preprocess_hash)` | `cache.make_key` |
| 6 | Plot interativo, com fallback para PNG em backend não-GUI | `plotting.show_or_save` |
| 7 | Último diretório persistido e oferecido como padrão | `config.resolve_startup` |

### Regra de invalidação

Alterar `directory`, `glob`, `pattern`, `metric_key`, `preprocess_steps`,
`allow_duplicates` ou `page_policy` limpa `records`, `discovery` e `report`.
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

## Adicionando uma métrica

Só `metrics.py` é tocado:

```python
@register(
    key="mean_std",
    name="Mean and standard deviation",
    description="Média e desvio-padrão amostral (ddof=1), em contagens brutas.",
    outputs=(
        OutputSpec("mean", "Mean", "counts"),
        OutputSpec("std", "Standard deviation", "counts"),
    ),
)
def _mean_std(image):
    return {"mean": float(image.mean()), "std": float(image.std(ddof=1))}
```

Como `outputs` é declarado estaticamente, a UI já lista as séries e oferece
`[s] select series` sem precisar rodar nada antes.

## Testes

```bash
python -m pytest
```

105 testes cobrindo parsing de nomes, duplicatas, política multi-página,
contrato das métricas, chave de cache, invalidação de estado, config corrompida,
CSV, fallback headless do gráfico e o loop de menu com teclas roteirizadas.

## Fora de escopo na v1

Passos de preprocessing com comportamento real (o pipeline existe, vazio), cache
persistente em disco, extração de dois parâmetros / séries agrupadas, varredura
recursiva de diretórios e formatos não-TIFF.
