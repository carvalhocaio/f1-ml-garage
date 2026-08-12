# 00 — Pipeline de Dados (FastF1)

## Por que FastF1, e não a API do Ergast/Jolpica?

O Ergast (e seu sucessor Jolpica) expõe resultados oficiais — grid, tempos
de volta agregados, classificação — mas não telemetria por volta (velocidade,
freada, marcha por canal de sensor). Como o Módulo 4 (não-supervisionado)
precisa de telemetria para clusterizar estilo de pilotagem, e o Módulo 2
precisa de contexto de volta (composto, idade do pneu, status de pista) para
prever tempo de volta, o FastF1 é a fonte que cobre os dois casos com um só
pipeline — sem precisar reconciliar dois schemas de fontes diferentes.

## Por que separar `laps.py` (puro) de `session.py` (rede)?

`normalize_laps` e `filter_accurate_laps` são funções puras: DataFrame
dentro, DataFrame fora, sem I/O. `session.py` é a única parte do projeto que
fala com a rede real da F1. Essa separação é o que torna a normalização
testável sem rede — os testes unitários constroem um DataFrame bruto no
schema do FastF1 na mão (`_raw_laps()`), sem precisar baixar uma sessão real.

O mesmo padrão aparece no `cotton-math-lab`: `hvi.py` gera dados sintéticos
(puro) para que módulos de estimação sejam testados sem depender de dados
reais da empresa. Aqui a motivação é ligeiramente diferente — não é sobre
"parâmetros conhecidos por construção", é sobre **velocidade e determinismo
dos testes**: rede é lenta, instável entre execuções, e a API da F1 não é
nossa para controlar.

## Por que `normalize_laps` não filtra nada?

Voltas de entrada/saída de pit, sob safety car, ou deletadas por comissários
continuam nas linhas de saída. A tentação seria já devolver "voltas limpas
prontas pro modelo" — mas diferentes módulos precisam de recortes diferentes
do mesmo dado:

- Um modelo de ritmo de corrida (Módulo 2) quer só voltas cronometradas
  confiáveis → `filter_accurate_laps`.
- Um modelo de pit stop (Módulo 3, DNF/estratégia) precisa justamente das
  voltas de entrada/saída que `filter_accurate_laps` descartaria.

Misturar normalização de schema com filtragem de negócio numa função só
forçaria a escolher um filtro "padrão" que inevitavelmente erra a mão para
algum módulo downstream. Separar as duas responsabilidades evita isso.

## Por que tempos em segundos (float) e não `Timedelta`?

`Timedelta` é a representação correta na fronteira com o FastF1, mas é
desconfortável para tudo que vem depois: `scikit-learn`, `numpy`, e qualquer
agregação estatística esperam `float`. Converter uma vez, na normalização, e
nunca mais tocar em `Timedelta` no resto do projeto, evita que cada módulo
downstream reimplemente a mesma conversão (e delegue paras si o
tratamento de `NaT`, que `.dt.total_seconds()` já resolve para `NaN`).

## Por que um marker `integration` separado, pulado por padrão?

Testes que batem na API real do FastF1 são lentos (podem levar minutos antes
do cache aquecer) e dependem de uma rede que não controlamos — não são o
tipo de teste que se quer rodando em todo `make test`. O padrão aqui é o
mesmo espírito do grupo `oracle` do `cotton-math-lab` (dependência opcional,
só roda quando pedido explicitamente), adaptado para "rede" em vez de
"biblioteca pesada": `make test` roda só os testes unitários; `make
test-integration` exige `F1_ML_GARAGE_RUN_INTEGRATION=1` de propósito, para
que rodar contra a API real seja sempre uma decisão explícita, nunca um
efeito colateral de rodar a suíte inteira.

## Decisões de design registradas

- **`is_pit_in_lap`/`is_pit_out_lap` como booleanos, não os timestamps
  brutos:** para o Módulo 1, o que importa é "esta volta é afetada por um
  pit stop?". O timestamp exato do evento de pit não é usado por nenhum
  módulo planejado; se isso mudar, é fácil adicionar depois sem quebrar o
  schema atual.
- **`compound` normalizado para minúsculas:** o FastF1 retorna
  `"SOFT"/"MEDIUM"/"HARD"` (maiúsculas); minúsculas evitam bugs bobos de
  comparação de string mais adiante (feature engineering, one-hot encoding).
- **`deleted` convertido para `pandas.BooleanDtype` (`"boolean"`):** o
  schema bruto documenta `Deleted` como `bool | None` — usar o tipo nullable
  do pandas em vez de `object` deixa explícito que é um booleano com
  ausência possível, e evita que `None` vire `NaN` (float) silenciosamente.

## Resultados (`results.py`)

Mesmo padrão de `laps.py`: `normalize_results` conhece o schema bruto de
`SessionResults`, converte `Q1`/`Q2`/`Q3`/`Time` de `Timedelta` para
segundos, e não filtra nenhum piloto (inclusive quem abandonou continua na
saída — descartar essas linhas destruiria justamente o sinal que um modelo
de DNF precisa aprender).

`dnf` é a única coluna genuinamente derivada do módulo. Em vez de aceitar o
critério do FastF1 (`fastf1.core.DriverResult.dnf`) como está, reimplementamos
vetorizado com regex (`^\+\d+\s*Laps?$`) em vez do slice fixo
`Status[3:6] == "Lap"` que a biblioteca usa internamente — esse slice
assume um único dígito no número de voltas de atraso e erra silenciosamente
para atrasos de 10 voltas ou mais (`"+10 Laps"[3:6]` é `" La"`, não
`"Lap"`, então o FastF1 classificaria esse piloto como DNF por engano).
Coberto por teste (`test_double_digit_lapped_driver_is_not_dnf`).

## Telemetria (`telemetry.py`)

Cobre o schema combinado de `session.laps.pick_drivers(...).get_telemetry()`
— canais de carro (velocidade, RPM, marcha, acelerador, freio, DRS) e de
posição (X/Y/Z, status em/fora de pista), uma linha por amostra de alta
frequência, não por volta.

Duas decisões que valem registro:

- **`Throttle == 104` vira `throttle_invalid=True` e `throttle_pct=NaN`,
  não é descartado.** O FastF1 documenta 104 como valor de erro/dado
  indisponível, distinto de um acelerador saturado em 100%. Mascarar sem
  descartar a linha preserva a amostra para outros canais (velocidade, RPM
  continuam válidos nesse instante) e ainda permite medir a taxa de erro
  por sessão, se algum módulo futuro quiser.
- **`X`/`Y`/`Z` convertidos de décimos de metro para metros
  (`POSITION_SCALE_TO_METERS = 0.1`).** É a unidade documentada pelo
  próprio FastF1, mas não-óbvia — deixar isso implícito faria qualquer
  cálculo de distância no Módulo 4 (clustering de estilo de pilotagem)
  silenciosamente errado por um fator de 10.

## Carregamento (`session.py`)

`load_session_results` e `load_driver_telemetry` seguem o mesmo formato de
`load_session_laps`: uma chamada fina ao FastF1, cache habilitado
externamente via `enable_cache()`. `load_driver_telemetry` usa
`Laps.pick_drivers` (não `pick_driver`, singular — depreciado desde a 3.1.0
do FastF1) para filtrar as voltas de um piloto antes de puxar a telemetria
combinada de todas elas.

## Utilitário compartilhado (`timeutils.py`)

`timedelta_to_seconds` começou como uma função privada dentro de `laps.py`.
Quando `results.py` e `telemetry.py` precisaram da mesma conversão, ela foi
extraída para `timeutils.py` — mesma lógica, um só lugar, evitando que os
três módulos de normalização divirjam sutilmente com o tempo (ex.: um
tratando `NaT` diferente dos outros).
