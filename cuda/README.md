![](../icone.png)
# Redes neurais e Deep Learning utilizando Java
[**Cleuton Sampaio**](https://github.com/cleuton) - [**LinkedIn**](https://www.linkedin.com/in/cleutonsampaio/) 

![](../88x31.png)

[**English version**](../english/gpu_cpu)

## Programação CUDA

Sei que é sobre **Java**, mas, para entender como funciona a programação **CUDA** para GPUs nVidia, é preciso mergulhar primeiro no C++. Vou mostrar um pequeno exemplo de programação paralela, para você entender as diferenças entre a programação sequencial da CPU e a paralela da GPU.

Na verdade, este exemplo é muito utilizado em redes neurais e **deep learning**: Produto de matrizes!

## Produto de matrizes

Em álgebra linear, o produto de duas matrizes, **A** e **B**, é obtido pela soma dos produtos de cada linha de **A** e cada coluna de **B**. Exemplo, o produto das matrizes: 

**A**:

```
-1  2  4
 0  5  3
 6  2  1
 ```
 **B**: 

 ```
 3  0  2
 3  4  5 
 4  7  2
 ```
Resultará na seguinte matriz: 

```
19	36	16
27	41	31
28	15	24
```

Como? Vejamos... Cada elemento de cada linha de **A** deve ser multiplicado por cada elemento de cada coluna de **B** e o produto somado será o elemento da nova linha da matriz resultante. Vejamos a multiplicação da primeira linha: 

```
A(1,1) * B(1,1) + A(1,2) * B(2,1) + A(1,3) * B(3,1) = 19
A(1,1) * B(1,2) + A(1,2) * B(2,2) + A(1,3) * B(3,2) = 36
A(1,1) * B(1,3) + A(1,2) * B(2,3) + A(1,3) * B(3,3) = 16
```

O resultado terá a forma: Linhas de **A** e colunas de **B**. Para que seja possível o produto, o número de colunas de **A** deve ser igual ao número de linhas de **B**.

## Programação GPU

Já mostramos programação de **GPU** no exemplo [**"GPU ou CPU?"**](../gpu_cpu), portanto, só vou falar aqui um pouco sobre como o **kernel** é executado pela GPU.

A GPU é composta de **cores** ou núcleos paralelos, cada um executando um **thread**. Os **threads** são organizados em **blocks** (blocos) e estes em **grids** (grade). Quando disparamos um **kernel** os vários **threads** o executarão em paralelo, portanto, não há como garantir a ordem de execução. 

Para multiplicar as matrizes **A** e **B** deste exemplo, podemos trabalhar com 9 threads, cada um processando a multiplicação de uma linha de **A** por uma coluna de **B**. Neste caso podemos pensar em cada **block** contendo 9 **threads** e cada **grid** contendo um **block**. Isso fica mais fácil de ver no exemplo [**matmul.cu**](./matmul.cu):

```
    // Blocks & grids:
    dim3 blocks(size,size);
    dim3 grid(1,1);

    // Call the kernel:
    matmul<<<grid,blocks>>>(gpu_A,gpu_B,gpu_C,size);
```
O **size** das minhas matrizes é 3, pois ambas são **3 x 3**. Neste trecho de código eu determinei a quantidade e a distribuição dos **threads** e **blocks** que eu usarei e estou invocando o meu **kernel**.

É confuso, eu sei, mas há um [**tutorial muito bom da nVidia**](https://www.tutorialspoint.com/cuda/cuda_quick_guide.htm) sobre o assunto.

As etapas da programação básica **CUDA** são estas:

1. Declare o seu **kernel**;
- *A função marcada com **__global__** que será executada pela GPU;*
2. Crie suas variáveis **CPU**;
- *Declare e aloque espaço para seus vetores na memória da CPU;*
3. Crie suas variáveis **GPU** e copie os valores;
- *Declare e aloque espaço na **GPU** para seus vetores, copiando os valores para lá;*
4. Determine o formato do **grid** e dos **blocks** de **threads**;
- *Calcule quantos **threads** serão necessários e arrume a distribuição em **blocks**;*
5. Invoque o **kernel**;
- *Chame sua função **kernel** passando as variáveis GPU;*
6. Copie o resultado de volta para a **CPU**;
- *Copie os dados do vetor calculado na **GPU** para a **CPU**;*
7. Libere a memória da **GPU**;
- *A memória da **GPU** é limitada, portanto, precisamos liberá-la.*

Veja o [**código-fonte**](./matmul.cu) e identifique todas essas etapas.

## Kernel

A multiplicação de matrizes na GPU é um processo diferente. Para começar, nós "desnormalizamos" as matrizes, criando vetores. Uma forma de fazer isso é **row first**. Por exemplo, pegamos a primeira matriz e colocamos cada linha depois da outra, "achatando" a estrutura: 

```
float cpu_A[] = {-1,2,4,0,5,3,6,2,1};
```

Depois, quando invocarmos o **kernel** precisaremos usar um *truque* para saber qual é a linha e qual é a coluna que queremos trabalhar: 

```
    // Row and Column indexes: 
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

```
Usamos os objetos da GPU **blockIdx**, **blockDim** e **threadIdx** para calcular a linha e a coluna da nossa matriz "achatada". Os dois objetos possuem valores **x** e **y**, correspondendo ao modelo de **grid** e **blocks** que passamos ao invocar o **kernel**.

Cada **thread** calculará um valor da matriz de resultado, portanto, precisa multiplicar uma linha por uma coluna: 

```
    // Are they bellow the maximum?
    if (col < size && row < size) {
       float result = 0;
       for(int ix=0;ix<size;ix++) {
          result += A[row*size+ix]*B[ix*size+col];
       }
       C[row*size+col] = result;
    }
```

O cuidado que tomei com o **if** foi para evitar que **threads** extras entrem no código, causando erro. 

O Kernel nada deve retornar e a comunicação deve ser sempre através das variáveis de GPU.

## Compilação e execução

Você pode compilar esse código com o **nvcc**: 

```
export PATH=$PATH:/usr/local/cuda/bin

nvcc -lcuda matmul.cu -o matmul
```

Muitas vezes o path do **cuda** já está setado, mas, em algumas plataformas, isso não ocorre, como no **nVidia Jetson Nano**.

Para executar: 

```
./matmul
```

E você verá o resultado na console: 

```
cleuton@cleuton:~/Documentos/projetos/cuda$ ./matmul
19	36	16
27	41	31
28	15	24
```


