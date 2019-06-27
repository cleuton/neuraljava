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
	private static final Logger log = Logger.getLogger(TrainModel.class.getName());
	public static void train(String imagePath, String modelPath, int numLabels, String testImage) throws IOException {
		
		// Random
		int seed = 42;
		Random random = new Random(seed);
		
		// Hiperparameters
		int height = 80;
		int width = 80;
		int channels = 1;
		int batchSize = 10;
		double trainTestRatio = 0.7;
		int iterations = 1;
		int epochs = 50;
		double learningRate = 0.01;
		
		// Prepare dataset:
		
        LabelGen labelMaker = new LabelGen();
        File mainPath = new File(imagePath);
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, random);
        int numExamples = Math.toIntExact(fileSplit.length());
        
        // Network configuration:
        
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
        MultiLayerNetwork network = new MultiLayerNetwork(conf);

        network.init();
```

Para inferir as classes a partir do nome do arquivo, tive que criar a classe [**LabelGen**](./src/main/java/com/neuraljava/demos/imageutil/LabelGen.java), que implementa a interface **PathLabelGenerator**.

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

1. Path das imagens originais;
2. Path onde as imagens processadas serão gravadas;
3. Path onde o modelo será salvo;
4. Path da imagem de teste;