package com.datalearninghub.samples.perceptron;

import java.util.Random;

/**
 * Implementação do algoritmo de redes neurais "Perceptron", desenvolvido por
 * Frank Rosenblatt. Serve para demonstrar o uso de redes neurais.
 * Esta classe pode atuar como classificador de elementos linearmente 
 * separáveis.
 * Um perceptron pode ter mais de uma variável de entrada, mas isto aumentaria a quantidade
 * de dimensões, tornando a visualização difícil. O único requisito é que, para efeito de 
 * classificação, os elementos sejam linearmente separáveis.
 * As variáveis principais são públicas, para efeito de demonstração.
 * @author Cleuton Sampaio
 *
 */
public class Perceptron {
	public double pesoA;
	public double pesoB;
	public double somaErros;
	public double bias;
	public double learningRate;
	public int    numeroIteracoes;  // Simplificando
	public double output;
	
	public Perceptron (int numeroIteracoes, double learningRate) {
		this.numeroIteracoes = numeroIteracoes;
		this.learningRate = learningRate;
		Random random = new Random();
		random.setSeed(System.currentTimeMillis());
		pesoA = random.nextDouble() * random.nextDouble()* random.nextDouble();
		pesoB = random.nextDouble() * random.nextDouble()* random.nextDouble();
		bias = random.nextDouble() * random.nextDouble()* random.nextDouble();
	}
	
	/**
	 * Treina a rede.
	 * @param X Matrix com 2 níveis: linha e coluna
	 * @param y Rótulo (0 ou 1)
	 * 	 
	 **/
	public void treinar(double [][] X,double [] y) {
		for (int iter=0; iter<this.numeroIteracoes;iter++) {
			somaErros = 0;
			for (int i=0; i<X.length;i++) {
				double erro = 0.0;
				double estimativa = classificar(X[i][0],X[i][1]);
				erro = y[i] - estimativa;
				pesoA = pesoA + learningRate * erro * X[i][0];
				pesoB = pesoB + learningRate * erro * X[i][1];
				bias  = bias + learningRate * erro;
				somaErros += Math.pow(erro, 2);
			}
			System.out.println("Iteração: " + (iter+1)
					+ ", RMSE: "+Math.sqrt(somaErros/X.length));
		}
	}
	
	/**
	 * Classifica uma instância
	 * @param a valor da variável (linha)
	 * @param b valor da variável (coluna)
	 * @return Rótulo estimado
	 */
	public double classificar(double a, double b) {
		double rotulo = 0.0;
		rotulo = sigmoid(a * pesoA + b * pesoB + bias);
		return rotulo;
	}
	
	public double sigmoid(double x) {
		return (1/( 1 + Math.pow(Math.E,(-1*x))));
	}
}
