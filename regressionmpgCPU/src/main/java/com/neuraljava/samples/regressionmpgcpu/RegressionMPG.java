/*
 
   Copyright 2018 Cleuton Sampaio

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. 
   
   Este trabalho é para demonstração de redes neurais e não tem objetivo 
   de desempenho ou precisão. O Autor não se responsabiliza pelo seu uso e
   não fornecerá suporte. 
 */
package com.neuraljava.samples.regressionmpgcpu;

import java.io.IOException;
import java.util.Arrays;

import org.datavec.api.records.Record;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.column.InvalidValueColumnCondition;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.util.logging.Logger;


public class RegressionMPG {
	private static final Logger log = Logger.getLogger(RegressionMPG.class.getName());
	private void modelBuild() throws IOException, InterruptedException {

		// Carrega o arquivo CSV diretamente do Resource: 
        RecordReader rr = new CSVRecordReader(1,',');
        rr.initialize(new FileSplit(new ClassPathResource("auto-mpg.csv").getFile()));	
        
        // Vamos criar o esquema para transformação do dataset: 
        Schema schema = new Schema.Builder()
        		.addColumnInteger("mpg")
                .addColumnCategorical("cylinder", Arrays.asList("3","4", "5","6","8"))
                .addColumnInteger("displacement")        
                .addColumnInteger("horsepower")
                .addColumnInteger("weight")
                .addColumnFloat("acceleration")
                .addColumnCategorical("modelyear",Arrays.asList("70","71","72","73",
                		"74","75","76","77","78","79","80","81","82"))   
                .addColumnCategorical("origin",Arrays.asList("1","2","3"))      
                .addColumnsString("carname")
                .build();
        
        // Temos que eliminar valores inválidos na coluna "horsepower" e converter os valores categóricos:
        TransformProcess transformProcess = new TransformProcess.Builder(schema)  
                .removeColumns("carname")    
                .categoricalToOneHot("cylinder")
                .categoricalToOneHot("modelyear")
                .categoricalToOneHot("origin")
                .filter(new InvalidValueColumnCondition("horsepower"))
                .build();
        
        RecordReader tRR = new TransformProcessRecordReader(rr,transformProcess); 
        
        // Agora, vamos criar um Iterator e separar os dados de treino e teste (80% e 20%): 
        DataSetIterator iter = new RecordReaderDataSetIterator.Builder(tRR, 8)
        	       .regression(0, 0)
        	       .build();
        
        DataSetIteratorSplitter splitter = new DataSetIteratorSplitter(iter,49,0.8);
        DataSetIterator treino = splitter.getTrainIterator();
        DataSetIterator test   = splitter.getTestIterator();
        
        // Criando a arquitetura da Rede:
        int seed = 124;
        int colunasInput = 25; // São 7, mas depois do "one-hot-encoding" viram 25
        int colunasOutput = 1;
        int nHidden1 = 10;
        int nHidden2 = 5;
        float learningRate = 0.01f;
        int epochs = 400;
        MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(colunasInput).nOut(nHidden1)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(nHidden1).nOut(nHidden2)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(nHidden2).nOut(colunasOutput).build())
                .build()
        );        
        net.setListeners(new ScoreIterationListener(8));
        
        // Treinamento da rede: 
        for( int i=0; i<epochs; i++ ){
        	System.out.println("****** Epoch: " + i);
        	treino.reset();
            net.fit(treino);
        }
        
        // Teste e avaliação:
        RegressionEvaluation eval = net.evaluateRegression(test);
        System.out.println(eval.stats());
        
        // Predições: 
        // Valor esperado: 18
        double [][] valores = new double[][] {{0, 0, 0, 1, 0, 250, 105, 3459, 16, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0}};
        INDArray vetor = Nd4j.create(valores);
        System.out.println("Esperdo: 18, previsto: " + net.output(vetor));
        /*
        int conta = 0;
        tRR.reset();
        while(tRR.hasNext()) {
        	Record r = tRR.nextRecord();
        	conta++;
        	System.out.println(r);
        }
        System.out.println(conta);
        */
	}

	public static void main(String[] args) throws IOException, InterruptedException {
		RegressionMPG rmpg = new RegressionMPG();
		rmpg.modelBuild();

	}

}