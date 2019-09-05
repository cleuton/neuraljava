![](../icone.png)
# Redes neurais e Deep Learning utilizando Java
[**Cleuton Sampaio**](https://github.com/cleuton) - [**LinkedIn**](https://www.linkedin.com/in/cleutonsampaio/) 

![](../88x31.png)

[**English version**](../english/cnn_java)

## Convolutional Neural Network

É uma rede neural dupla, sendo que a primeira parte é dedicada a capturar as características de cada imagem, enquanto a segunda, classifica essa característica de acordo com os pesos.

Aqui está [**um artigo**](http://www.obomprogramador.com/2019/03/reconhecimento-e-classificacao-facial.html) no qual explico tudo sobre **CNN** com exemplos em Python e **Tensorflow**.

## Implementação em Java

Podemos implementar qualquer tipo de rede neural em Java, inclusive **CNN**. As redes CNN exigem pré-processamento das imagens, de modo a aumentar a precisão do reconhecimento. 

Nesta demonstração, usaremos os pacotes: 

- [**OpenIMAJ**](http://openimaj.org/): Para alinhamento das faces e pré-processamento das imagens;
- [**Deeplearning4J**](https://deeplearning4j.org/): Para criação do modelo de rede neural.

## Banco de imagens

O [**Labeled Faces in the Wild**](http://vis-www.cs.umass.edu/lfw/) é um banco de fotos de figuras públicas, que pode ser utilizado para treinar uma CNN. 

Precisamos pré-processar as imagens, extraindo e alinhando os rostos e convertendo para tons de cinza (1 canal de cor). Exemplo: 

Imagem original:

![](./cleuton_001.jpg)

Imagem preparada:

![](./cleuton_0010.jpg)

Eu baixei o database LFW e extraí as imagens, mudando um pouco a estrutura. Em vez de pastas com o nome das pessoas, eu juntei todos os arquivos e pego o nome a partir do path dos arquivos.

## Preprocessamento de imagens

A classe [**PrepareFaces.java**](./src/main/java/com/neuraljava/demos/imageutil/PrepareFaces.java) possui o método **getFaces()** que utiliza as classes do **OpenIMAJ** para detectar, extrair e alinhar os rostos, criando novas imagens.

```
public class PrepareFaces {
	public static List<BufferedImage> getFaces(String imagePath) throws IOException {
		List<BufferedImage> facesList = new ArrayList<BufferedImage>();
		File arq = new File(imagePath);
		BufferedImage imagem = ImageIO.read(arq);
		FImage fimage = ImageUtilities.createFImage(imagem);
		FKEFaceDetector detector = new FKEFaceDetector(1);
		List<KEDetectedFace> faces = detector.detectFaces(fimage);	
		RotateScaleAligner aligner = new RotateScaleAligner();
		for (KEDetectedFace face : faces) {
			FImage aligned = aligner.align(face);
			BufferedImage bfi = ImageUtilities.createBufferedImage(aligned);
			facesList.add(bfi);
		}
		return facesList;
	}
	
	public static Set<String> prepareImages(String sourcePath, String targetPath) throws IOException {
		Set<String> names = new HashSet<String>();
		File[] files = new File(sourcePath).listFiles();
	    for (File file : files) {
	    	String name = getNameFromFile(file);
	    	names.add(name);
	    	List<BufferedImage> faces = getFaces(file.getAbsolutePath());
	    	int faceCount = 0;
	    	for (BufferedImage img : faces) {
	    		int pos = file.getName().indexOf('.');
	    		String fileName = file.getName().substring(0,pos) + faceCount + ".jpg";
	    		File outputfile = new File(targetPath + "/" +  fileName);
	    		ImageIO.write(img, "jpg", outputfile);
	    		faceCount++;
	    	}
	        System.out.println("Name: " + name);
	    }
	    return names;
	}

```

## Criação do modelo

Utilizando o **Deeplearning4J** criamos o modelo de rede neural na classe [**TrainModel**](../src/main/java/com/neuraljava/demos/imageutil/TrainModel.java): 

```
		// Random
		int seed = 42;
		Random random = new Random(seed);
		
		// Hiperparametros
		
		int height = 80;
		int width = 80;
		int channels = 1;
		int batchSize = 10;
		double trainTestRatio = 0.6;
		int iterations = 1;
		int epochs = 500;
		double learningRate = 0.01;
		
		// Preparar dataset com o nosso label generator:
		
        LabelGen labelMaker = new LabelGen();
        File mainPath = new File(imagePath);
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, random);
        int numExamples = Math.toIntExact(fileSplit.length());
        BalancedPathFilter pathFilter = new BalancedPathFilter(random, labelMaker, numExamples, numLabels, 0);
        
        // Separação treino e teste:
        
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, trainTestRatio, 1 - trainTestRatio);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];
        
        // Configuração da rede:
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .stride(1,1)
                        .nOut(32)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(1,1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(height,width,1)) 
                .build();
```

Para inferir as classes a partir do nome do arquivo, tive que criar a classe [**LabelGen**](./src/main/java/com/neuraljava/demos/imageutil/LabelGen.java), que implementa a interface **PathLabelGenerator**.

É preciso preparar os dados para alimentar o modelo: 

```
        // Carga dos dados:
        
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        
        // Iterador de treino:
        
        ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator trainIter;
        trainRR.initialize(trainData, null);
        trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, numLabels);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);
        
        // Iterador de teste:
        
        ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
        testRR.initialize(testData);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, numLabels);
        scaler.fit(testIter);
        testIter.setPreProcessor(scaler);

```

E, finalmente, eu uso o conceito de **Early Stopping** para gerar o modelo de melhor desempenho: 

```
        // Configuração de Early Stopping:
        
        int maxEpochsWithNoImprovement = 10;
        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
        		.epochTerminationConditions(new MaxEpochsTerminationCondition(epochs),new ScoreImprovementEpochTerminationCondition(maxEpochsWithNoImprovement) )
        		.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
        		.scoreCalculator(new DataSetLossCalculator(testIter, true))
                .evaluateEveryNEpochs(1)
        		.modelSaver(new LocalFileModelSaver(modelPath))
        		.build();
        
        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,conf,trainIter);

        // Treinamento da rede: 

        EarlyStoppingResult result = trainer.fit();
        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());
        
        // Recupera o modelo com melhor desempenho:
        
        MultiLayerNetwork net = (MultiLayerNetwork) result.getBestModel();  
        
        // Predições:

        trainIter.reset();
        DataSet testDataSet = trainIter.next();
        List<String> allClassLabels = trainRR.getLabels();
        int labelIndex = testDataSet.getLabels().argMax(1).getInt(0);
        int[] predictedClasses = net.predict(testDataSet.getFeatures());
        String expectedResult = allClassLabels.get(labelIndex);
        String modelPrediction = allClassLabels.get(predictedClasses[0]);
        System.out.print("\nNome da pessoa: " + expectedResult + " predição: " + modelPrediction + "\n\n");
```

## Uso de GPU

Esta demonstração utiliza a **GPU** para executar o treinamento da rede neural. Para isto, algumas coisas deve ser acrescentadas no **pom.xml**: 

```
	<dependency>
		<groupId>org.nd4j</groupId>
		<artifactId>nd4j-cuda-10.1</artifactId>
		<version>1.0.0-beta4</version>
	</dependency>
	<dependency>
		<groupId>org.deeplearning4j</groupId>
		<artifactId>deeplearning4j-cuda-10.0</artifactId>
		<version>1.0.0-beta4</version>
	</dependency>	
```

## Para executar o código

Ao executar o comando ```mvn clean package``` será criado um **uber jar** contendo todas as classes. Basta executá-lo passando os argumentos: 

1. args 0: Path das imagens originais;
2. args 1: Path para gravar as imagens processadas;
3. args 2: Path para salvar o modelo.