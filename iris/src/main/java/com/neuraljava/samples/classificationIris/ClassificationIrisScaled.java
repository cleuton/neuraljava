package com.neuraljava.samples.classificationIris;

import java.io.IOException;
import java.util.Arrays;
import java.util.logging.Logger;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;



public class ClassificationIrisScaled {
	private static final Logger log = Logger.getLogger(ClassificationIrisScaled.class.getName());
	private void modelBuild() throws IOException, InterruptedException {
		
		// Carrega o arquivo CSV diretamente do Resource: 
        RecordReader rr = new CSVRecordReader(0,','); // Não há header, logo, não vamos pular a primeira linha!
        rr.initialize(new FileSplit(new ClassPathResource("iris.data").getFile()));	
        
        // Vamos criar o esquema para transformação do dataset: 
        Schema schema = new Schema.Builder()
        		.addColumnFloat("sepal_length")
        		.addColumnFloat("sepal_width")
        		.addColumnFloat("petal_length")
        		.addColumnFloat("petal_width")
                .addColumnCategorical("class", Arrays.asList("Iris-setosa","Iris-versicolor", "Iris-virginica"))
                .build();
        
        // Transformando atributos discretos em one-hot-encodeds:
        TransformProcess transformProcess = new TransformProcess.Builder(schema)  
                .categoricalToInteger("class")
                .build();
        
        RecordReader tRR = new TransformProcessRecordReader(rr,transformProcess); 
        int labelIndex = 4;  // O índice do label é 4 (começa de zero)
        int numClasses = 3;     
        int batchSize = 150;    

        DataSetIterator iterator = new RecordReaderDataSetIterator(tRR,batchSize,labelIndex,numClasses);
        DataSet allData = iterator.next();
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);  //Use 65% of data for training

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();    
        
        // Padronizando os dados: 
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);  // Utilizamos as estatísticas dos dados de treino
        INDArray stats = ((NormalizerStandardize) normalizer).getMean();
        INDArray devs  = ((NormalizerStandardize) normalizer).getStd();
        normalizer.transform(trainingData); 
        normalizer.transform(testData); 
        
        // Criando a arquitetura da Rede:
        int seed = 124;
        int colunasInput = 4; 
        int colunasOutput = 3;
        int nHidden = 8;
        float learningRate = 0.01f;
        int epochs = 500;
        
        MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(colunasInput).nOut(nHidden)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT) // Multiclass Cross Entropy
                        .activation(Activation.SOFTMAX)
                        .nIn(nHidden).nOut(colunasOutput).build())
                .build()
        );        
        
        net.setListeners(new ScoreIterationListener(8));
        
        // Treinamento da rede: 
        for( int i=0; i<epochs; i++ ){
        	System.out.println("****** Epoch: " + i);
            net.fit(trainingData);
        }
        
        // Teste e avaliação:
        Evaluation eval = new Evaluation(3);
        INDArray output = net.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output);
        log.info(eval.stats());       

        // Predição: 
        // Valor esperado: 1 Iris-Versicolor
        
        double [][] valores = new double[][] {{5.61, 2.88, 4.0, 1.1}};
        INDArray vetor = Nd4j.create(valores);
        
        // Padronizando: 
        DataNormalization norm2 = new NormalizerStandardize(stats,devs);
        norm2.transform(vetor);

        System.out.println("Esperado: 1, previsto: " + net.output(vetor));
        
        /*
        int conta = 0;
        tRR.reset();
        while(tRR.hasNext()) {
        	conta++;
        	Record r = tRR.nextRecord();
        	System.out.println(r);
        }
        System.out.println(conta);
        */
        
	}
	
	public static void main(String[] args) throws IOException, InterruptedException {
		ClassificationIrisScaled cIris = new ClassificationIrisScaled();
		cIris.modelBuild();

	}
}
