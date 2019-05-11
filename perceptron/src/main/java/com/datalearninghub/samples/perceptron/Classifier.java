package com.datalearninghub.samples.perceptron;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;

/**
 * Usa o Perceptron para classificar as amostras de um dataset com 2 variáveis.
 * @author Cleuton Sampaio
 *
 */
public class Classifier {

	class Datasetds {
		double [][] valores;
		double [] rotulo;
	}
	
	Perceptron perceptron;

	void processa() throws Exception {
		treinar();
		testar();
	}
	
	void treinar() throws Exception {
		Datasetds treino = preprocessar("dados.csv",149);
		perceptron = new Perceptron(10,0.2);
		perceptron.treinar(treino.valores, treino.rotulo);
	}

	void testar() throws Exception {
		Datasetds teste = preprocessar("dados_teste.csv",91);
		int erros = 0;
		for(int i=0; i<teste.valores.length;i++) {
			double especie = Math.round(perceptron.classificar(teste.valores[i][0], teste.valores[i][1]));
			if(especie != teste.rotulo[i]) {
				erros++;
			}
			System.out.println("x1:" + teste.valores[i][0] 
					+ " x2: " + teste.valores[i][1] 
					+ " Origem: " + teste.rotulo[i]
					+ " Estimado: " + especie);
		}
		System.out.println("Percentual de erros: " + (((double)erros / teste.valores.length)*100) + "%");
	}

	Datasetds preprocessar(String nome, int numElementos) throws Exception {
		Datasetds Dataset = new Datasetds();
		Dataset.valores = new double[numElementos][2];
		Dataset.rotulo = new double[numElementos];
		InputStream in = this.getClass().getResourceAsStream("/" + nome); 
		BufferedReader reader = new BufferedReader(new InputStreamReader(in));
		String linha = null;
		linha = reader.readLine(); // ignora rótulos
		int i = 0;
		while ((linha = reader.readLine()) != null) {
			if(linha.length()<2) {
				break;
			}
			String [] vetor = linha.split(",");
			Dataset.valores[i][0] = Double.parseDouble(vetor[0]);
			Dataset.valores[i][1] = Double.parseDouble(vetor[1]);
			Dataset.rotulo[i] = Double.parseDouble(vetor[2]);
			i++;
		}
		return Dataset;		
	}




	public static void main(String[] args) throws Exception {
		(new Classifier()).processa();
	}
}
