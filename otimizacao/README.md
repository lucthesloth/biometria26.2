# Tentativa de otimização da ConvNeXt-Tiny

Objetivo: reduzir a taxa de falsa aceitação (FMR) e a de falsa rejeição (FNMR) da ConvNeXt-Tiny. As outras redes
não entram. Os resultados principais do projeto continuam os da configuração base; esta pasta registra o que foi
tentado, o que ficou e o que foi descartado. Houve duas etapas, e a segunda tem seção própria mais abaixo. O
melhor sistema soma 12 redes e chega a EER de 2,74% na FVC2002 DB1, 3,33% nos 30 dedos de teste da FVC2004 DB1 e
3,82% no protocolo oficial, contra 7,87%, 8,32% e 10,42% da base.

Regra de seleção da primeira etapa: cada escolha usa só a EER nos 20 dedos de validação da FVC2002 DB1. O limiar de cada
configuração é o da EER nessa validação, congelado para os três testes. Os testes aparecem nas tabelas para todas
as tentativas, mas não decidiram nenhuma delas.

## Primeira etapa: configuração base e configuração otimizada

| Item | Base | Otimizada |
|---|---|---|
| Rede | ConvNeXt-Tiny, ImageNet-1k, `features.0-3` congeladas | a mesma |
| Dados de treino | 60 dedos da FVC2002 DB1 (480 imagens) | 390 dedos: os mesmos 60 da DB1 mais os 110 de cada uma das bases DB2, DB3 e DB4 da FVC2002 (3.120 imagens) |
| Validação | 20 dedos da FVC2002 DB1 | os mesmos |
| Épocas | 80 | 40 (a melhor época ficou perto da 17) |
| Perda, otimizador, aumento | ArcFace (m = 0,3, s = 30), AdamW lr 3e-4, rotação 30°, translação 10%, blur | os mesmos |
| Entrada | uma rede, imagem com realce de Gabor | duas redes: uma com realce de Gabor, outra só com a normalização de Hong |
| Score | cosseno | soma ponderada dos dois cossenos, cada um normalizado pelos impostores de validação; peso escolhido na validação |
| Limiar | EER na validação, congelado | o mesmo procedimento |
| Tempo de treino | 5,5 min | 11 min por rede, duas redes |
| Inferência | uma rede, 9 ms, 112 MB | duas redes, cerca de 18 ms, 224 MB |

Nesta etapa a FVC2004 continua fora de todo treino. As bases DB2 a DB4 da FVC2002 usam outros sensores e outros voluntários,
e os 30 dedos de teste da DB1 seguem fora do treino. A configuração otimizada deixa de treinar "só na FVC2002
DB1", e por isso fica separada dos resultados principais.

## Resultados

Média ± desvio de três sementes de treino (2026, 1999 e 1234), taxas em %, FMR e FNMR no limiar congelado.
Valores completos em `results/otimizacao/summary.csv`.

| Configuração | Avaliação | EER | FMR | FNMR | TAR@FAR=1% |
|---|---|---|---|---|---|
| Base (60 dedos, Gabor) | Validação | 3,84 ± 0,44 | 3,84 | 3,93 | 83,9 |
| | FVC2002 DB1, teste | 7,87 ± 0,29 | 6,39 ± 1,02 | 9,56 ± 1,34 | 68,8 |
| | FVC2004 DB1, teste | 8,32 ± 0,44 | 6,92 ± 1,06 | 10,79 ± 2,65 | 46,5 |
| | FVC2004 DB1-A, oficial | 10,42 ± 0,17 | 7,57 ± 1,07 | 14,10 ± 1,70 | 50,9 |
| Base, Gabor + Hong | Validação | 2,93 ± 0,30 | 2,93 | 3,04 | 90,8 |
| | FVC2002 DB1, teste | 5,78 ± 0,33 | 6,48 ± 0,41 | 5,40 ± 0,79 | 81,3 |
| | FVC2004 DB1, teste | 7,23 ± 0,17 | 7,69 ± 1,29 | 6,59 ± 1,55 | 48,5 |
| | FVC2004 DB1-A, oficial | 9,12 ± 0,15 | 8,79 ± 0,68 | 9,56 ± 0,86 | 47,3 |
| 390 dedos, Gabor | Validação | 2,27 ± 0,38 | 2,26 | 2,32 | 93,9 |
| | FVC2002 DB1, teste | 5,17 ± 0,50 | 4,24 ± 1,05 | 6,47 ± 1,40 | 85,2 |
| | FVC2004 DB1, teste | 6,55 ± 1,29 | 3,81 ± 0,68 | 11,35 ± 4,95 | 72,2 |
| | FVC2004 DB1-A, oficial | 7,26 ± 1,00 | 3,89 ± 0,52 | 12,30 ± 3,56 | 70,1 |
| **390 dedos, Gabor + Hong (otimizada)** | Validação | 1,86 ± 0,42 | 1,86 | 1,90 | 96,1 |
| | FVC2002 DB1, teste | **4,05 ± 0,36** | **3,56 ± 0,74** | **4,56 ± 0,25** | 89,0 |
| | FVC2004 DB1, teste | **5,48 ± 0,72** | **3,69 ± 0,75** | **8,81 ± 3,74** | 74,0 |
| | FVC2004 DB1-A, oficial | **6,23 ± 0,49** | **3,52 ± 0,75** | **10,95 ± 2,72** | 70,8 |

Diferença da configuração otimizada para a base, com intervalo de 95% por bootstrap pareado sobre dedos
(`results/otimizacao/bootstrap_ci.csv`, 2.000 réplicas):

| Avaliação | EER | FMR | FNMR |
|---|---|---|---|
| FVC2002 DB1, teste | -3,82 (-5,31 a -2,34) | -2,83 (-5,50 a -0,52) | -5,00 (-7,46 a -2,82) |
| FVC2004 DB1, teste | -2,85 (-4,73 a -1,54) | -3,22 (-5,86 a -1,51) | -1,98 (-5,08 a 1,07) |
| FVC2004 DB1-A, oficial | -4,19 (-5,24 a -3,18) | -4,05 (-5,81 a -2,68) | -3,14 (-4,86 a -1,40) |

Oito das nove diferenças têm intervalo que exclui zero. A exceção é a FNMR nos 30 dedos de teste da FVC2004 DB1.

O que cada mudança faz:

- Mais dados de treino reduzem sobretudo a FMR (de 6,9% para 3,8% na FVC2004 DB1) e quase não mexem na FNMR
  dessa base.
- A soma das redes com e sem realce reduz sobretudo a FNMR (de 9,6% para 5,4% na FVC2002 DB1 com os dados base) e
  não reduz a FMR.
- As duas juntas reduzem as duas taxas.

A FNMR na FVC2004 DB1 ainda varia muito entre sementes (desvio de 3,7 pontos). Na rede com realce treinada com
390 dedos, o limiar congelado foi de 0,27 a 0,37 conforme a semente.

## Tentativas descartadas na primeira etapa

Uma mudança por vez sobre a configuração base, semente 2026. A base com essa semente tem EER de 3,41% na
validação e 8,18%, 8,34% e 10,46% nos testes.

| Tentativa | Mudança | Validação | FVC2002 | FVC2004 | Oficial | Decisão |
|---|---|---|---|---|---|---|
| `erase` | apagamento aleatório, p = 0,5 | 3,04 | 6,81 | 11,13 | 11,32 | descartada |
| `pose` | apagamento e translação de 20% | 2,60 | 7,26 | 10,46 | 11,39 | descartada |
| `unfreeze` | congelar só até `features.1` | 3,21 | 7,70 | 8,94 | 11,25 | descartada |
| `margin05` | margem ArcFace de 0,5 | 3,75 | 7,74 | 8,88 | 10,67 | descartada |
| `e160` | 160 épocas | 3,29 | 8,10 | 7,69 | 10,44 | descartada |
| `pool_tta` | média do embedding em rotações de -10°, 0° e 10° (sobre 390 dedos) | 3,21 contra 2,70 sem ela | 5,43 | 6,44 | 6,61 | descartada |

Quatro das cinco primeiras melhoram a validação em 0,1 a 0,8 ponto, e `margin05` a piora. `pose` junta duas
mudanças (apagamento e translação). Com 20 dedos de validação e uma semente, esses ganhos estão dentro do ruído
(o desvio entre sementes da base é 0,44), e nenhuma foi levada às três sementes. Várias baixaram a EER de teste
na FVC2002 DB1, mas o teste não entra na decisão. Duas delas
pioram a FVC2004 DB1 em 2 a 3 pontos, sinal de que o ganho na validação era ajuste ao sensor da DB1. A média em
rotações piorou a validação e saiu por essa regra, embora os testes tenham melhorado um pouco.

## Segunda etapa: mais bases, validação com outros sensores e mais redes

A primeira etapa (acima) deixou a FNMR na FVC2004 DB1 entre 9% e 11%. A segunda mexe em quatro coisas.

| Item | Etapa 1 | Etapa 2 |
|---|---|---|
| Dados de treino | 390 dedos da FVC2002 | 745 dedos: os 390 mais FVC2004 DB2 (90), DB3 (90) e DB4 (110) e a U.are.U da Neurotechnology (65) |
| Validação | 20 dedos da FVC2002 DB1 | os mesmos 20 mais 20 dedos da FVC2004 DB2 e 20 da DB3, fora do treino (`protocols/wide_val`); impostores só dentro de cada base |
| Épocas | 40 | 30 |
| Redes por semente | duas (Gabor, Hong) | quatro: Gabor e Hong, cada uma com rotação de 30° e translação de 10% ou com 45° e 20% |
| Score de cada rede | cosseno | cosseno normalizado contra os 100 dedos de treino mais próximos de cada imagem (centroides), nos dois sentidos |
| Fusão | peso escolhido na validação | média simples dos scores z-normalizados |
| Sistema final | uma semente (média de três nas tabelas) | as três sementes somadas: 12 redes, cerca de 110 ms e 1,3 GB |

A FVC2004 DB1 continua fora de todo treino e de toda validação. As bases DB2 a DB4 da FVC2004 usam outros sensores
e outros voluntários que a DB1, mas seguem o mesmo roteiro de coleta, com dedos distorcidos, secos e úmidos. A
validação precisou mudar porque os 20 dedos da FVC2002 DB1 não têm esse tipo de captura e não conseguem dizer se
uma mudança ajuda nela. O limiar da EER na validação nova fica perto de 0,25 em cosseno, contra 0,36 na antiga.

Resultados (`results/otimizacao/summary.csv`, linhas `wide_*`; FMR e FNMR no limiar congelado):

| Configuração | Validação | FVC2002 DB1, teste | FVC2004 DB1, teste | FVC2004 DB1-A, oficial |
|---|---|---|---|---|
| Base (60 dedos, Gabor) | 3,84 (regra antiga) | 7,87 / 6,39 / 9,56 | 8,32 / 6,92 / 10,79 | 10,42 / 7,57 / 14,10 |
| Etapa 1 (390 dedos, 2 redes) | 1,86 (regra antiga) | 4,05 / 3,56 / 4,56 | 5,48 / 3,69 / 8,81 | 6,23 / 3,52 / 10,95 |
| 745 dedos, 2 redes (`wide_gh`) | 3,25 | 3,80 / 3,75 / 3,85 | 4,26 / 2,60 / 7,54 | 4,68 / 2,49 / 8,75 |
| 745 dedos, 4 redes (`wide_4`) | 2,96 | 3,28 / 3,37 / 3,37 | 3,96 / 2,25 / 7,66 | 4,35 / 2,26 / 8,69 |
| 745 dedos, 4 redes, score normalizado (`wide_4_as`) | 2,67 | 3,07 / 3,15 / 3,06 | 3,81 / 2,35 / 6,55 | 4,30 / 2,25 / 8,05 |
| **745 dedos, 12 redes, score normalizado (`wide_12_as`)** | 2,26 | **2,74 / 2,75 / 2,74** | **3,33 / 2,06 / 6,43** | **3,82 / 1,88 / 7,50** |

Cada célula de teste é EER / FMR / FNMR em %. As linhas com 2 e 4 redes são médias de três sementes; a de 12 redes
é um sistema só. As EER de validação das duas regras não se comparam entre si, porque a validação nova tem bases
mais difíceis.

Diferença do sistema de 12 redes para a base, com intervalo de 95% (`results/otimizacao/bootstrap_ci.csv`):

| Avaliação | EER | FMR | FNMR |
|---|---|---|---|
| FVC2002 DB1, teste | -5,13 (-7,00 a -3,45) | -3,64 (-6,69 a -1,22) | -6,83 (-10,12 a -3,85) |
| FVC2004 DB1, teste | -4,99 (-6,90 a -3,52) | -4,86 (-7,52 a -3,16) | -4,37 (-8,33 a -0,39) |
| FVC2004 DB1-A, oficial | -6,60 (-7,80 a -5,35) | -5,69 (-7,47 a -4,25) | -6,60 (-8,91 a -4,46) |

As nove diferenças excluem zero. A FNMR na FVC2004 DB1 segue a maior taxa. Os pares genuínos rejeitados têm pouca
área em comum (ponta do dedo contra o núcleo, ou rotação grande), e um embedding global lida mal com isso.

Tentativas da segunda etapa, uma semente (2026), com a decisão pela validação da vez:

| Tentativa | Sobre | Validação | Decisão |
|---|---|---|---|
| `pool_aug`: distorção suave da pele e afinamento ou engrossamento das cristas, na GPU | 390 dedos | 3,45 contra 2,70 | descartada |
| `pool_ema`: média móvel dos pesos (0,995) | 390 dedos | 3,04 contra 2,70 | descartada |
| Busca de rotação na comparação, até ±30° | 390 dedos, três sementes | 2,32 contra 2,27 | descartada |
| Distorção elástica do torchvision | 390 dedos | interrompida: 89 s por época na CPU | descartada |
| `w2_pose3`: translação de 30% | 745 dedos | 5,71 contra 5,18 | descartada |
| `w2_pose`: rotação de 45° e translação de 20% | 745 dedos | 5,18 contra 5,78 | aceita, como segunda rede de cada entrada |
| Score normalizado contra os dedos de treino | 745 dedos, 2 redes | 3,10 contra 3,39 | aceita |
| Soma das três sementes | 390 dedos | 1,83 contra 1,86 | aceita |
| Busca de translação na comparação (9 deslocamentos de 40 px) | 785 dedos, 1 rede | 1,96 contra 2,20 | não incluída: nove passagens por imagem e não foi repetida nas redes finais |

As linhas `wide` e `wide_hong` do `summary.csv` são um treino anterior com os 785 dedos e a validação antiga, de
uma semente, que serviu para medir o efeito dos dados (EER de 5,74% para 3,39% na FVC2002 DB1 com uma rede).

Seis dos doze treinos finais rodaram na RTX 4060 local e seis numa A100 do Colab (semente 1234 e parte da 1999),
então repetir tudo numa máquina só pode dar números um pouco diferentes.

## Como reproduzir

Da raiz do repositório, com `data/raw/FVC2002/Dbs` completo e o pipeline principal já executado:

```bash
PY=.venv/Scripts/python.exe
$PY otimizacao/make_pool_protocol.py
for N in gabor hong; do $PY preprocess.py --dataset fvc2002_pool --set preprocess.normalization=$N; done
for SEED in 2026 1999 1234; do
  bash otimizacao/run_variant.sh pool $SEED gabor fvc2002_pool --set train.epochs=40
  bash otimizacao/run_variant.sh pool_hong $SEED hong fvc2002_pool --set train.epochs=40
done
$PY otimizacao/ensemble.py --name pool_gabor_hong --a results/otimizacao:pool --b results/otimizacao:pool_hong
$PY otimizacao/ensemble.py --name base_gabor_hong --a results:convnext_tiny_gabor --b results:convnext_tiny_hong
$PY otimizacao/table.py
$PY otimizacao/bootstrap.py
```

Cada treino com 390 dedos leva cerca de 11 minutos numa RTX 4060. As tentativas descartadas usam
`run_variant.sh <nome> 2026 gabor fvc2002_db1 --set <chave>=<valor>`; `extract_tta.py` gera os embeddings com
média em rotações.

Segunda etapa, com `data/raw/FVC2004/Dbs` e `data/raw/Neurotechnology/UareU`:

```bash
$PY otimizacao/make_pool_protocol.py --dataset wide_pool --holdout 20
for N in gabor hong; do $PY preprocess.py --dataset wide_pool --set preprocess.normalization=$N; done
export PROTO=wide_val
POSE="--set train.augment.rotation_deg=45 --set train.augment.translate=0.2"
for SEED in 2026 1999 1234; do
  bash otimizacao/run_variant.sh w2 $SEED gabor wide_pool --set train.epochs=30
  bash otimizacao/run_variant.sh w2_hong $SEED hong wide_pool --set train.epochs=30
  bash otimizacao/run_variant.sh w2_pose $SEED gabor wide_pool --set train.epochs=30 $POSE
  bash otimizacao/run_variant.sh w2_pose_hong $SEED hong wide_pool --set train.epochs=30 $POSE
done
M="w2 w2_hong w2_pose w2_pose_hong"
$PY otimizacao/fuse.py --name wide_gh --members w2 w2_hong
$PY otimizacao/fuse.py --name wide_4 --members $M
$PY otimizacao/fuse.py --name wide_4_as --members $M --asnorm 100
$PY otimizacao/fuse.py --name wide_12_as --members $M --asnorm 100 --pool-seeds
$PY otimizacao/table.py
$PY otimizacao/bootstrap.py
```

Cada treino com 745 dedos leva cerca de 17 minutos numa RTX 4060. `run_variant.sh` também grava, em
`results/otimizacao/<avaliação>/<rede>/`, cada rede sozinha com o limiar da validação antiga. `show.py` compara
essas execuções. Os scores das redes `w2*` sozinhas não estão versionados: `results/otimizacao/` guarda
só as somas (`wide_gh`, `wide_4`, `wide_12` e as versões `_as`, com score normalizado).

## Limitações

- A validação tem 20 dedos e escolhe a época, o peso da fusão e o limiar.
- O ganho com mais dados não separa o efeito de ter mais dedos do efeito de ter mais sensores.
- As tentativas descartadas rodaram com uma semente só.
- A validação da segunda etapa tem 60 dedos, e a decisão de mudá-la veio depois de olhar os erros da FVC2004 DB1
  de uma execução. Nenhum dedo da FVC2004 DB1 entra nela, mas a escolha não foi cega.
- A segunda etapa treina com a FVC2004 DB2 a DB4, colhidas com o mesmo roteiro da DB1. O resultado na FVC2004 DB1
  deixa de medir a troca para uma coleta de outro tipo, e passa a medir a troca de sensor e de voluntários.
- O sistema de 12 redes custa 12 vezes a inferência da base.
- Os testes já tinham sido vistos nos resultados da configuração base antes desta tentativa. As escolhas daqui
  seguiram a validação, mas a decisão de tentar veio depois de conhecer aqueles números.
