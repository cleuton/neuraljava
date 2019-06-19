![](../icone.png)

# Neural Networks and Deep Learning using Java

[**Cleuton Sampaio**](https://github.com/cleuton) - [**LinkedIn**](https://www.linkedin.com/in/cleutonsampaio/) 

![](../88x31.png)

## GPU vs CPU?

[**Original source code here**](../../gpu_cpu)

A big question for the developers of Machine Learning and Deep Learning applications is whether or not to use a GPU, after all, GPUs are still very expensive. To get an idea, let's see the price of a typical GPU for processing AI in Brazil: something between $ 1,000.00 and $ 7,000.00 (or more).

The purpose of this tutorial is to demonstrate the need to use GPU for **Deep Learning** processing.

**Amazon** has appropriate instances for GPU processing. Here's how to compare prices in two configurations:

| Type | vCPUs | GPUs | RAM | hourly cost (US$) |
| --- | --- | --- | --- | --- |
| c5.2xlarge | 8 | 0 | 16 GiB |	0.34 |
| p3.2xlarge | 8 | 1 | 61 GiB | 3.06 |
| p3.8xlarge | 32 | 4 | 244 GiB | 12.24 |


Anyone who has ever trained a Machine or Deep Learning model knows that using GPU can shorten training time from days / hours to minutes / seconds, right?

But, is it really necessary? Can we not use a **cluster** of cheap machines, as we do with **Bigdata**?

The simplest and most direct answer is: **YES**, GPUs are required to train models and **nothing will replace them**. However, you need to to know hwo to program with GPUs in order to get the most out of them, ant not all libraries and frameworks do this effectively.

## How a GPU works (for Dummies)

Let's start with an analogy adapted from what I have seen on a [Data Science Academy](http://www.datascienceacademy.com.br) course,  and I really enjoyed it.

Imagine a fast motorcycle, like ... 1000 CC ... maybe a **Kawazaki**. It's a very fast motorcycle, right? Now imagine that you have 8 of these motorcycles and you do pizza delivery. Each bike can take 1 order to the customer, so if there are more than 8 orders, someone will have to wait for one of the motorcycles to be available for delivery.

This is how the CPU works: Very fast and focused on sequential processing. Each core is a very fast motorcycle. Of course you can adapt so that each motorcycle delivers more than one pizza at a time, but in any case, it will be sequential processing: Deliver one pizza, deliver the next, and so on.

Now, let's think you have 2000 bicycles and 2000 deliver boys. Although the motorcycles are still much faster, you have many more bicycles and can deliver multiple orders at once, avoiding service queues. The slowness of bicycles is compensated by parallelism.

So that's how the GPU works: Oriented for parallel processing.

If we compare the processing time of each task, the CPU wins, but if we consider the parallelism, the GPU is unbeatable. This is why it is used for intensive processing tasks and calculations, such as: Virtual coin mining and Deep Learning.

## How can we program for GPU?

 GPU Programming is not simple. To begin with, you have to consider that there is more than one GPU vendor and there are two best-known programming frameworks:

- **CUDA**: Compute Unified Device Architecture, for **Nvidia** chipsets;
- **OpenCL**: for other vendors, like: **AMD**.

The CUDA programming interface is made in C, but there are *bindings* for **Python**, such as [**PyCuda**](https://documen.tician.de/pycuda/) and for **Java**, such as [**JCuda**](http://www.jcuda.org/). But they are a little more difficult to learn and program.

And you need to fully understand the [**CUDA**](https://developer.nvidia.com/cuda-zone) platform, as well as its individual components, such as [**cuDNN**](https://developer.nvidia.com/cudnn) or the [**cuBLAS**](https://docs.nvidia.com/cuda/cublas/index.html).

However, there are easier and more interesting alternatives that use the GPU, such as [**Deeplearning4J**](https://deeplearning4j.org/docs/latest/nd4j-overview) and its associated project, [**ND4J**](https://deeplearning4j.org/docs/latest/nd4j-overview). The **ND4J** project is like the **numpy** of Java, but supercharged with **steroids**! It allows you to use the available GPU(s) in a simple and practical way, and this is what we will use in this tutorial.

## Before starting

You must have an **NVidia GPU** in your computer, with the appropriate drivers installed. Find out what the GPU you have. Then make sure you have installed the [**correct driver from Nvidia**](https://www.nvidia.com/download/index.aspx?lang=en). Then install the [**CUDA Toolkit**](https://www.nvidia.com/Download/index.aspx?lang=br). If everything is correct, you can execute the command below:

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

## AI jobs

What is an AI job? **Deep Learning** is an AI job! It is based on two complex mathematical operations:

- **Feedforward**: Basically the linear combination of the weight matrices with the values ​​in each layer, from the input to the end;
- **Backpropagation**: Differential calculation of each gradient of each neuron (including BIAS), from the last layer to the beginning, in order to adjust the weights.

**Feedforward** is repeated for each record in the input set and multiplied by the number of iterations or **epochs** we wish to train, that is, many times. And **Backpropagation** can be done at the same frequency, or at regular intervals, depending on the learning algorithm used.

In summary: Vector and differential calculations of multiple simultaneous values.

This is why GPUs are required for development, training and also for inferences, depending on the complexity of the model.

## Demonstration

*(This demo was written in Portuguese)*

This tutorial's project is a **Java** application that performs aa matrix multiplication (scalar product), a common operation in deep learning jobs. It multiplies the matrices only once, first on the CPU, then on the GPU (using **ND4J** and **CUDA Toolkit**). Note that it is not even a model of deep learning, but only a single basic operation.

The file [**pom.xml**](../../gpu_cpu/pom.xml) configures **ND4J** to use the GPU with the **CUDA** platform:
```
	<dependency>
	    <groupId>org.nd4j</groupId>
	    <artifactId>nd4j-cuda-10.1</artifactId>
	    <version>1.0.0-beta4</version>
	</dependency>
```

The main class [**MatMul**](../../gpu_cpu/src/main/java/com/neuraljava/demos/gpu/MatMul.java) is a simple application that defines two matrices and calculates their dot product, first in the CPU, then on the GPU, using the **ND4J** library.

I'm working with 2 matrices of 500 x 300 and 300 x 400, something a typical neural network must have.

My laptop is an **I7, eighth generation**, and has a chipset **Nvidia MX110**, which is well *entry level* with 256 cores and Cuda Capability 5, a very simple machine... One **K80 board** has more than 3,500 cuda cores and cuda capability 8 or higher.

Let's look at the program execution:
*(Messages are in portuguese, but I will provide translation)*

```
CPU Interativo 	(nanossegundos): 111.203.589

...
1759 [main] INFO org.nd4j.linalg.jcublas.ops.executioner.CudaExecutioner  - Device Name: [GeForce MX110]; CC: [5.0]; Total/free memory: [2101870592]


GPU Paralelo 	(nanossegundos): 9.905.426


Percentual de tempo no cálculo da GPU com relação à CPU: 8.907469704057842
```

First line says that CPU calculation took 111,203,589 nanoseconds, and two last lines says that GPU calculation took 9,905,426, or 8.9% of CPU calculation.

## Conclusion

On an entry level GPU like mine, the scalar matrix product ran on the GPU in only 8.9% of the time it ran on the CPU. An abyssal difference. Check it out:
- CPU time: **111,203,589** nanoseconds;
- GPU time: **9,905,426** nanoseconds.

Considering that the scalar product is just ONE operation, and that Feedforward does this operation thousands of times, it is reasonable to believe that this difference should be much greater if we were actually training a neural network.

And it doesn't help using **clusters** of CPUs, **RDMA** etc, because nothing, NOTHING is able to match the performance of a single GPU. Except more than one GPU, of course!

Well, I hope I have demonstrated here two things: **GPU is essential** and how we can use it directly from a **Java** application. If you want, you can even convert that model of [**MLP**](../../multilayerperceptron)  we have created to run on the **GPU** and write a shiny and clever post.
