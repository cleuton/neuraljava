![](../../icone.png)

# Neural Networks and Deep Learning using Java

[**Cleuton Sampaio**](https://github.com/cleuton) - [**LinkedIn**](https://www.linkedin.com/in/cleutonsampaio/) 

![](../../88x31.png)

## Convolutional Neural Network

[**Original source code here**](../../cnn_java)

It is a double Neural Network model composed by some convolutional layers (not all neurons are connected), dedicated to extract image features, and some dense connected layers, which classify the image by its features.

[**Here you have my post**](https://github.com/cleuton/facerec_cnn) explining how a **CNN** works.

## Java implementation

We can create any neural network model with Java, including **CNN**. A CNN requires images pre-processing in order to raise the accuracy.

In this demo we'll use: 

- [**OpenIMAJ**](http://openimaj.org/): Image pre-processing and face alignment;
- [**Deeplearning4J**](https://deeplearning4j.org/): To create the model.

## Image database

The [**Labeled Faces in the Wild**](http://vis-www.cs.umass.edu/lfw/) is a public image database that can be used to train a CNN.

We need to pre-process the imagens, extracting and aligning the faces, and also converting them to grayscale. For example:emplo: 

Original image:

![](../../cnn_java/cleuton_001.jpg)

Prepared image:

![](../../cnn_java/cleuton_0010.jpg)

## Image preprocessing

Class [**PrepareFaces.java**](../../cnn_java/src/main/java/com/neuraljava/demos/imageutil/PrepareFaces.java) has method **getFaces()**, which uses  **OpenIMAJ** to detect, extract and align faces, creating new images.

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

## Model creation

I created the model using **Deeplearning4J** in class [**TrainModel**](../../cnn_java/src/main/java/com/neuraljava/demos/imageutil/TrainModel.java): 

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

To get the classes from the file name, I had to create class [**LabelGen**](../../cnn_java/src/main/java/com/neuraljava/demos/imageutil/LabelGen.java), implementing interface **PathLabelGenerator**.

## GPU usage

This code uses the **GPU** to train the Neural Network. To use the GPU I had to include this in the **pom.xml**: 

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

## How to run the code

When you run a  ```mvn clean package``` it will create an **uber jar**. Just run it with the arguments: 

1. raw images path
2. processed images path
3. model save path
4. test image path