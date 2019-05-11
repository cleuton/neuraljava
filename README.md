![](./icone.png)
# Redes neurais e Deep Learning utilizando Java
[**Cleuton Sampaio**](https://github.com/cleuton) - [**LinkedIn**](https://www.linkedin.com/in/cleutonsampaio/) 

![](./88x31.png)

Todo o conteúdo, quando não explicitamente indicado, está liberado sob a [licença Creative Commons Atribuição 4.0 Internacional](http://creativecommons.org/licenses/by/4.0/). O código-fonte está liberado sob a [licença Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

# Introdução

## Papo furado

Sério, o que sua empresa roda no ambiente de produção? De acordo com a popularidade e as estatísticas, é mais provável que sejam sistemas feitos em **Java**, certo? Claro que sim. Java é o novo **COBOL**. Embora muitos sistemas corporativos e comerciais estejam sendo desenvolvidos em outras linguagens, como **python**, os dados demonstram que Java ainda é a plataforma campeã no mundo corporativo.

E existem razões para isso, por exemplo, o grande número de desenvolvedores Java no mundo. Mas não é só isso... Java é uma plataforma completa, madura e baseada em padrões, com um vasto ecossistema de componentes e frameworks igualmente maduros. Além disso, temos a [performance *acachapante*](https://benchmarksgame-team.pages.debian.net/benchmarksgame/faster/python.html) de Java sobre outras linguagens, como o python. Eu até [**já escrevi um artigo sobre este assunto, no meu outro blog: O Bom programador**](http://www.obomprogramador.com/2019/03/python-paralelismo-e-gil-nem-tudo.html).

![](./benchmarks.png)

Porém, linguagens de script, como python e **R** são muito utilizadas em estudos de ciência de dados, machine learning e **deep learning**, talvez pela facilidade de codificação, ou pelo excelente conjunto de bibliotecas para isto. Realmente, criar um modelo baseado em Redes Neurais em Python ou R, seja utilizando o **Keras** ou outra API, é muito simples. Você só precisa saber COMO criar redes neurais, a implementação é muito enxuta, com baixa [**complexidade acidental**](http://www.obomprogramador.com/2012/12/complexidade-acidental.html). Isso é totalmente o contrário de Java!

Mas dizer que Java não serve para deep learning ou ciência de dados, é simplesmente **papo furado**. 

Eu diria que a melhor linguagem para se trabalhar com qualquer problema de **Inteligência Artificial** (incluíndo ciência de dados) seria **C++**. Por quê? Bom, as bibliotecas foram escritas em C++, e o desempenho é imbatível! Mas C++ é muito difícil de programar e até de compilar! Requer muito tempo e esforço para construir modelos simples. Logo, Java pode ser uma alternativa mais fácil e popular.

Resolvi começar esta série de artigos porque fui convidado a lecionar um curso sobre implementação de soluções de **IA** utilizando Java, e quero compartilhar com vocês a minha experiência. 

*Bom proveito!*

## Inteligência artificial

Desde os anos 50, quando um cientista chamado [*John McCarthy* usou este termo pela primeira vez](https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial), a **IA** tem sido alvo de muitos estudos e tendências, especialmente agora, nos últimos 10 anos. 

O estudo de IA tem muitas vertentes, como por exemplo a construção de [**Agentes inteligentes**](https://pt.wikipedia.org/wiki/Agente_inteligente). Uma corrente de estudos focou em criação de [**Sistemas especialistas**](http://www.scielo.br/scielo.php?script=sci_arttext&pid=S0100-19651997000100006), utilizando *motores de inferência*, baseados em regras que utilizam técnicas como **encadeamento para a frente** de modo a concluir sobre dados novos. Eles são capazes de armazenar novas regras e interligá-las de modo a tomar decisões. 

Porém, a especificação de regras requer conhecimento do domínio do problema, portanto, não é tarefa simples. Outra corrente de IA se baseou nos próprios dados, imitando a forma como o nosso cérebro *aprende*, e esta é a corrente que conhecemos como **Redes Neurais**, e seu uso em aprendizagem de máquina é conhecido como *aprendizagem profunda* ou **Deep Learning**.

## Redes neurais

Deep learning (ou aprendizagem profunda) é um ramo da Data Science, mais especificamente de Machine learning, que utiliza grafos de funções em camadas para resolver problemas de classificação e regressão, supervisionados ou não. Estes grafos em camadas são também conhecidos pela metáfora: “Rede neural”,
pois se parecem com a arquitetura do cérebro humano.

![](./redeneural.png)

Os nós são o que chamamos de **neurônios** e as arestas são os dados e os pesos (veremos mais adiante). Cada neurônio recebe as entradas, multiplica pelos pesos e passa tudo por uma **função de ativação**, que determinará o sinal de saída.

![](./neuronio.png)

## Deep Learning

Hoje, este termo (*deep learning*) vem ganhando popularidade a cada dia. Em Português significa: "Aprendizagem profunda", que é basicamente utilizar redes neurais para criar modelos de predição (regressão, classificação ou agrupamento). Na Wikipedia temos uma boa definição: 

"
*A aprendizagem profunda, do inglês Deep Learning (também conhecida como aprendizado estruturado profundo, aprendizado hierárquico ou aprendizado de máquina profundo) é um ramo de aprendizado de máquina (Machine Learning) baseado em um conjunto de algoritmos que tentam modelar abstrações de alto nível de dados usando um grafo profundo com várias camadas de processamento, compostas de várias transformações lineares e não lineares.*
"

**Como funciona?**

Quando usamos *deep learning* queremos criar um modelo preditivo, ou seja, um algoritmo capaz de prever algum resultado, com base em variáveis coletadas. Pode ser prever o valor de um imóvel, com base em sua localização e tamanho, ou pode ser classificar uma imagem como sendo um cão, um gato ou uma colher.

Para que o modelo funcione, precisamos *treiná-lo*. Ele precisa "aprender" a associar as variáveis de entrada com uma determinada saída. Uma vez que ele tenha "aprendido" esta associação, podemos usá-lo para fazer **predições** (**inferências**). É claro que, para isto, precisamos salvar o "conhecimento" aprendido para reintroduzi-lo no modelo posteriormente.

Para entender como funciona esse processo de aprendizado, podemos usar um modelo bem simples de rede neural: o **Perceptron** que, na verdade, só tem um nó ou *neurônio*. 

Se o problema for mais complexo, podemos necessitar de um modelo de rede neural mais complexo, como o **Multi Layer Perceptron - MLP** (Perceptron com múltiplas camadas), formado por vários nós organizados em camadas. Na verdade, um MLP é um grafo, onde os nós são transformações e as arestas são valores.

## Perceptron

![](./percep.png)

Na figura, vemos um pereptron típico, com um só nó (*neurônio*). Ele possui nós de entrada, que nada processam. Neste caso, ele recebe duas variáveis, representadas pelos nós **a1** e **a2**, executa uma combinação linear com os pesos (**w1** e **w2**) e usa uma **função de ativação** para calcular qual é a saída **z** gerada. 

Você deve notar que existe uma terceira entrada que apenas tem o valor **1**. É o **bias** (ou **viés**). Ele serve de coeficiente linear da operação, deslocando da raiz. Mesmo que as entradas sejam zeros, a rede será capaz de aprender alguma coisa. O bias é multiplicado pelo seu peso (**bw**) e somado com as entradas x pesos, sendo alimentado no perceptron.

Inicialmente, todos os pesos são gerados aleatóriamente (entre -1 e 1, entre 0 e 1, entre -2 e 2 etc), portando, as primeiras previsões serão incorretas.A saída gerada, **z**, é o nosso valor alvo (ou *target*). Podemos compará-lo com o valor real e quantificar o **erro** que cometemos, portanto, podemos ajustar os pesos para tentar novamente. E fazemos diversas vezes, até que estejamos satisfeitos. 

Para treinar um perceptron, temos algums **parâmetros** a ajustar:
- Quantidade de entradas: (**i**) Quantas variáveis de entrada teremos;
- Quantidade de dados: (**n**) Quantos conjuntos de variáveis teremos para treinar e para testar;
- Taxa de aprendizado: **Learning hate** o quanto vamos ajustar os pesos a cada erro descoberto;
- Número de iterações: (**epochs**) quantas vezes vamos repetir o treinamento com todos os dados de treino;
- Função de ativação: Qual a função de ativação que vamos usar para gerar a saída do nó;
- Função de custo: Qual a função que queremos minimizar com o treinamento.

**Learning hate** é um parâmetro importante, pois indica o quanto o modelo vai "aprender" a cada erro. Mas, o que é "aprender"? Qual é o objetivo do treinamento? Ai entra a **função de custo**. Em um perceptron, a função de custo pode ser simples como esta: 

```
erro = t - z
```
Ou seja **t** que é o valor real ou **target** menos **z** que é a saída produzida pelo perceptron.

Precisamos otimizar essa função, procurando o seu mínimo e fazemos ajustes nos pesos para cada erro encontrado. Abra o projeto [**perceptron**](./perceptron) e execute. Você verá algo assim: 

```
Iteração: 8, RMSE: 0.08451685277702517
Iteração: 9, RMSE: 0.08038599611566254
Iteração: 10, RMSE: 0.07697726935792881
x1:1.01071991464072 x2: 1.14385421069991 Origem: 1.0 Estimado: 1.0
x1:0.074131851523044 x2: 0.123206823070496 Origem: 0.0 Estimado: 0.0
x1:1.961556880148 x2: 1.46033212028283 Origem: 1.0 Estimado: 1.0
x1:1.71230468169387 x2: 1.96311939860278 Origem: 1.0 Estimado: 1.0
x1:1.70525169301648 x2: 1.30856225370596 Origem: 1.0 Estimado: 1.0
x1:0.393956672046951 x2: 0.540630320013019 Origem: 0.0 Estimado: 0.0
x1:0.299135720184961 x2: 0.843663552563487 Origem: 0.0 Estimado: 0.0
x1:0.965832666407979 x2: 0.92372791146796 Origem: 0.0 Estimado: 0.0
```
Ele **converge** muito rapidamente, chegando a taxas de erro bem pequenas. **Convergir** é o fato do modelo "aprender" a associar entradas com a saída.

**Limitações**

Perceptrons tem a limitação de apenas trabalharem com linearidade, ou seja, no caso de classificação, eles conseguem separar dados que são linearmente separáveis, como os utilizados na demonstração:

![](./linearmenteseparavel.png)

Na figura vemos que há duas classes de elementos que podem ser separados apenas por uma reta. 

E se tivermos dados não linearmente separáveis? Um exemplo simples de entender é a função **XOR**:

x | y | XOR
--- | --- | ---
0 | 1 | 1
1 | 0 | 1
0 | 0 | 0 
1 | 1 | 0

Por exemplo, veja o gráfico da função **XOR** note que temos duas classes de elementos: Os que retornam **1** e os que retornam **zero**: 

![](./nao-linearmente-separavel.png)

A única maneira de separá-los seria traçar duas linhas. 

Outro exemplo de associação não linear: 

![](./non-linear.png)

Um perceptron não vai conseguir convergir com dados como estes. Para isto, precisamos usar mais neurônios, organizados em camadas. Aí entra o **MLP**.

## Multi Layer Perceptron 

Um **MLP** é um modelo de rede com vários nós (*neurônios*), organizados em camadas distintas: 

![](./mlp.png)
