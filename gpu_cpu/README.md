![](../icone.png)
# Redes neurais e Deep Learning utilizando Java
[**Cleuton Sampaio**](https://github.com/cleuton) - [**LinkedIn**](https://www.linkedin.com/in/cleutonsampaio/) 

![](../88x31.png)

[**English version**](../english/gpu_cpu)

## GPU ou CPU?

Uma grande dúvida dos desenolvedores de aplicações de Machine Learning e de Deep Learning é sobre a necessidade ou não de usar um computador com GPU, afinal de contas, GPUs ainda são muito caras. Para ter uma ideia, veja o preço de uma GPU típica para processamento de IA no Brasil custa algo entre US$ 1.000,00 e US$ 7.000,00 (ou mais). 

O objetivo deste tutorial é demonstrar a necessidade do uso de GPU para processamento de **Deep Learning**.

A Amazon possui instâncias apropriadas para processamento com GPU. Veja só a comparação de preços de duas configurações: 

| Instância | vCPUs | GPUs | RAM | Preço por hora (US$) |
| --- | --- | --- | --- | --- |
| c5.2xlarge | 8 | 0 | 16 GiB |	0,34 |
| p3.2xlarge | 8 | 1 | 61 GiB | 3,06 |
| p3.8xlarge | 32 | 4 | 244 GiB | 12,24 |

Qualquer um que já tenha treinado um modelo de Machine ou Deep Learning, sabe que o uso de GPU pode diminuir o tempo de treinamento de dias / horas para minutos / segundos, certo?

Mas, é realmente necessário? Não podemos usar um **cluster** de máquinas baratas, como fazemos com **Bigdata**?

A resposta mais simples e direta é: **SIM**, GPUs são necessárias para treinar modelos e nada as substituirá. Porém, você tem que programar direito, de modo a obter o melhor proveito do uso de **GPU**, e nem todas as bibliotecas e frameworks fazem isso com eficiência. 

## Como funciona a GPU

Vamos começar por uma analogia, adaptada do que eu vi no treinamento da [**Data Science Academy**](http://www.datascienceacademy.com.br) e que gostei muito. 

Imagine uma moto sinistra, tipo... 1000 CC... Sei lá, uma Kawazaki. É uma moto bem veloz, certo? Agora, imagine que você tenha 8 dessas motos e faz entrega de pizzas. Cada moto pode levar 1 pedido até o cliente, logo, se houver mais de 8 pedidos, alguém terá que esperar uma das motos ficar disponível para entrega. 

É assim que funciona a CPU: Muito rápida e voltada para processamento sequencial. Cada núcleo é uma moto muito veloz. É claro que você pode adaptar para que cada moto entregue mais de uma pizza por vez, mas, de qualquer forma, será um processamento sequencial: Entrega uma pizza, entrega a próxima etc.

Agora, vamos pensar que você tenha 2000 bicicletas e 2000 entregadores. Embora as motos sejam muito mais rápidas, você tem muito mais bicicletas e pode entregar vários pedidos de uma só vez, evitando filas de atendimento. A lentidão das bikes é compensada pelo paralelismo.

Assim é a GPU: Orientada para processamento em paralelo. 

Se compararmos o tempo de processamento de tarefas, a CPU ganha, mas, se considerarmos o paralelismo, a GPU é imbatível. É por isso que é utilizada para tarefas de processamento intensivo e cálculos, como: Mineração de moedas virtuais e Deep Learning.

## Como podemos programar para a GPU

A programação para GPU não é simples. Para começar, você tem que considerar que existe mais de um fornecedor de GPUs e que há dois frameworks de programação mais conhecidos: 

- **CUDA**: Compute Unified Device Architecture, dos chips **Nvidia**;
- **OpenCL**: Utilizado em GPUs de outros fornecedores, como a **AMD**.

A interface de programação CUDA é feita em C, mas existem *bindings* para **Python**, como o [**PyCuda**](https://documen.tician.de/pycuda/) e para **Java**, como o [**JCuda**](http://www.jcuda.org/). Mas são um pouco mais difíceis de aprender e programar. 

E você precisa entender bem da plataforma [**CUDA**](https://developer.nvidia.com/cuda-zone), assim como de seus componentes individuais, como o [**cuDNN**](https://developer.nvidia.com/cudnn) ou o [**cuBLAS**](https://docs.nvidia.com/cuda/cublas/index.html).

Porém, há alternativas mais fáceis e interessantes que utilizam a GPU, como o [**Deeplearning4J**](https://deeplearning4j.org/docs/latest/nd4j-overview) e seu projeto associado, o [**ND4J**](https://deeplearning4j.org/docs/latest/nd4j-overview). O **ND4J** é como se fosse o **numpy** do Java, só que com **esteróides**! Ele é capaz de sozinho permitir que você use a(s) GPU(s) disponíveis de maneira simples e prática, e é o que vamos usar neste tutorial. 

## Antes de mais nada

Você deve ter uma GPU **NVidia** em seu equipamento, com os drivers apropriados instalados. Procure saber qual é a GPU que você possui. Depois, certifique-se de haver instalado o [**driver correto, da Nvidia**](https://www.nvidia.com.br/Download/index.aspx?lang=br). Depois, instale o [**CUDA Toolkit**](https://www.nvidia.com.br/Download/index.aspx?lang=br). Se tudo estiver correto, você poderá executar o comando abaixo: 

```
nvidia-smi

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.39       Driver Version: 418.39       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce MX110       On   | 00000000:01:00.0 Off |                  N/A |
| N/A   50C    P0    N/A /  N/A |    666MiB /  2004MiB |      4%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1078      G   /usr/lib/xorg/Xorg                           302MiB |
|    0      1979      G   /usr/bin/gnome-shell                         125MiB |
|    0      2422      G   ...quest-channel-token=7684432457067004974   108MiB |
|    0     19488      G   ...-token=7864D1BD51E7DFBD5D19F40F0E37669D    47MiB |
|    0     20879      G   ...-token=8B052333281BD2F7FF0CBFF6F185BA98     1MiB |
|    0     24967      G   ...-token=62FCB4B2D2AE1DC66B4AF1A0693122BE    40MiB |
|    0     25379      G   ...equest-channel-token=587023958284958671    35MiB |
+-----------------------------------------------------------------------------+

```

## Trabalhos de IA

O que é um trabalho de IA? De Deep Learning? É baseado em duas operações matemáticas complexas: 

- **Feedforward**: Basicamente a combinação linear das matrizes de pesos com os valores em cada camada, da entrada até o final;
- **Backpropagation**: Cálculo diferencial de cada gradiente de cada neurônio (incluindo os BIAS), da última camada até o início, de modo a ajustar os pesos.
O Feedforward é repetido para cada registro do conjunto de entrada e multiplicado pela quantidade de iterações ou epochs que desejamos treinar, ou seja, muitas vezes. E o Backpropagation pode ser feito na mesma frequência, ou a intervalos regulares, dependendo o algoritmo de aprendizado utilizado. 

Em resumo: Cálculos vetoriais e diferenciais de múltiplos valores simultâneos. 

É por isto que as GPUs são necessárias para desenvolvimento, treinamento e também para inferências, dependendo da complexidade do modelo.

## Demonstração

O projeto deste tutorial é uma aplicação **Java** que executa uma multiplicação de matrizes (produto escalar), uma operação comum em trabalhos de deep learning. Ele multiplica as matrizes uma única vez, primeiro na CPU, depois na GPU (utilizando ND4J e CUDA Toolkit). Notem que nem chega a ser um modelo de machine learning, mas apenas uma única operação básica.

O arquivo [**pom.xml**](./pom.xml) configura o **ND4J** para usar a GPU com a plataforma **CUDA**: 

```
	<dependency>
	    <groupId>org.nd4j</groupId>
	    <artifactId>nd4j-cuda-10.1</artifactId>
	    <version>1.0.0-beta4</version>
	</dependency>
```

A classe principal [**MatMul**](./src/main/java/com/neuraljava/demos/gpu/MatMul.java) é uma aplicação simples, que define duas matrizes e calcula seu produto escalar, primeiramente na CPU, depois na GPU, utilizando o **ND4J**.

Estou trabalhando com 2 matrizes de 500 x 300 e 300 x 400, nada demais para uma rede neural típica. 

Meu laptop é um **I7, oitava geração**, e tem um chipset **Nvidia MX110**, que é bem "entry level", com 256 cuda cores e Cuda Capability 5, ou seja, nada demais... Uma placa K80 tem mais de 3.500 cuda cores e cuda capability 8 ou superior. 

Vejamos a execução do programa: 

```
CPU Interativo 	(nanossegundos): 111.203.589

...
1759 [main] INFO org.nd4j.linalg.jcublas.ops.executioner.CudaExecutioner  - Device Name: [GeForce MX110]; CC: [5.0]; Total/free memory: [2101870592]


GPU Paralelo 	(nanossegundos): 9.905.426


Percentual de tempo no cálculo da GPU com relação à CPU: 8.907469704057842
```

## Conclusão

Em uma GPU *entry level* como a minha, o produto escalar de matrizes rodou na GPU em apenas 8,9% do tempo que rodou na CPU. Uma diferença abissal. Vejam só: 
- Tempo CPU: **111.203.589** nanossegundos;
- Tempo GPU: **9.905.426** nanossegundos.
Considerando que o produto escalar é apenas UMA operação, e que o Feedforward envolve milhares de vezes esta operação, é razoável acreditar que essa diferença deve ser muito maior, se realmente estivéssemos treinando uma rede neural. 

E não adianta cluster, nem RDMA, pois nada, NADA é capaz de igualar o desempenho de uma única GPU. 

Bom, espero ter demonstrado aqui duas coisas: **GPU é essencial** e como podemos utilizá-la diretamente de uma aplicação **Java**. Se você quiser, pode até converter aquele modelo de [**MLP** que fizemos](../multilayerperceptron) para rodar na **GPU** e tirar uma tremenda onda. 
