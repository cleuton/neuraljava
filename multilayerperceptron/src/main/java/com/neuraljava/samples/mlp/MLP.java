package com.neuraljava.samples.mlp;

import java.util.Random;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * Multi Layer perceptron.
 * Este código implementa um MLP (aproximador universal) simples, com o propósito de demonstrar Redes Neurais multicamadas.
 * Há só um nó na camada de saída.
 * @author Cleuton Sampaio
 *
 */
public class MLP {
	public RealVector input;
	public RealMatrix hidden;
	public RealVector hiddenValues;
	public RealVector biasHidden;
	public RealVector output;
	public double outputValue;
	public double biasOutputWeight;
	public double learningHate;
	public int iterations;
	public double MSE;
	private Random random = new Random();

	public MLP() {
		double [][] wMatrix = new double [2][2];
		wMatrix[0] = random.doubles(2,-2,2.01).toArray();
		wMatrix[1] = random.doubles(2,-2,2.01).toArray();
		this.hidden = MatrixUtils.createRealMatrix(wMatrix);
		this.biasHidden = MatrixUtils.createRealVector(random.doubles(2,-2,2.01).toArray());
		this.output = MatrixUtils.createRealVector(random.doubles(2,-2,2.01).toArray());
		biasOutputWeight = random.doubles(-2, 2.01).findFirst().getAsDouble();
		this.learningHate = 0.2;
		this.iterations = 100;
	}
	
	/**
	 * Ativação da Rede com um só conjunto de dados de entrada. Usa Sigmoid como função de ativação
	 * @param x double[] vetor de características de um registro.
	 */
	public void forwardPropagation(double [] x) {
		this.input = MatrixUtils.createRealVector(x);
		hiddenValues = hidden.operate(input).add(biasHidden).map(v -> sigmoid(v));
		outputValue  = sigmoid(output.dotProduct(hiddenValues) + biasOutputWeight);
	}
	
	/**
	 * Backpropagation usando Gradient Descent
	 * @param y double saída esperada de um exemplo
	 * @param deltaz2 
	 */
	public void backPropagation(double t) {
		double z = this.outputValue;
		double erro = Math.pow((t-z), 2);
		double deltaz = (z - t) * z * (1 - z);
		this.MSE += erro;
		for(int i=0;i<2;i++) {
			this.output.setEntry(i, 
					this.output.getEntry(i) 
					- this.learningHate 
					* deltaz
					* this.hiddenValues.getEntry(i)				
					);
		}
		this.biasOutputWeight = this.biasOutputWeight 
								- this.learningHate * deltaz;
		for(int i=0;i<2;i++) {
			for(int j=0;j<2;j++) {
				this.hidden.setEntry(i, j, 
						this.hidden.getEntry(i, i) 
						- this.learningHate
						* deltaz
						* this.output.getEntry(j)
						* this.hiddenValues.getEntry(j)
						* (1 - this.hiddenValues.getEntry(j))
						* this.input.getEntry(i)
						);
			}
		}
		for(int i=0;i<2;i++) {
			this.biasHidden.setEntry(i, 
					this.biasHidden.getEntry(i)
					- this.learningHate
					* deltaz
					* this.output.getEntry(i)
					* this.hiddenValues.getEntry(i)
					* (1 - this.hiddenValues.getEntry(i))
					);
		}
		
	}

	private double derivPesosBiasHidden(double t, double z, int i) {
		double deriv = (z-t) * z * (1 - z) 
				* this.output.getEntry(i)
				* this.hiddenValues.getEntry(i)
				* (1 - this.hiddenValues.getEntry(i));
		return deriv;
	}

	private double derivPesosHidden(double t, double z, int i, int j) {
		double deriv = (z-t) * z * (1 - z) * this.output.getEntry(j) 
				* this.hiddenValues.getEntry(j) * (1 - this.hiddenValues.getEntry(j))
				* this.input.getEntry(i);
		return deriv;
	}

	private double derivPesosBiasOutput(double t, double z) {
		return (z-t) * z * (1 - z);
	}

	private double derivPesosOutput(double t, double z, int i) {
		double deriv = (z-t) * z * (1 - z) * this.hiddenValues.getEntry(i);
		return deriv;
	}

	public void treinar(double [][] X, double [] y) {
		for(int it=0; it<this.iterations; it++) {
			this.MSE = 0.0;
			for(int n=0;n<X.length;n++) {
				forwardPropagation(X[n]);
				this.backPropagation(y[n]);
			}		
			this.MSE /= X.length;
			System.out.println("Fim iteração: " + (it+1) + " MSE: " + this.MSE);
		}
	}
	
	private void printAll(String string) {
		System.out.println(string);
		System.out.println("hidden: " + this.hidden);
		System.out.println("hiddenValues: " + this.hiddenValues);
		System.out.println("biasHidden: " + this.biasHidden);
		System.out.println("output: " + this.output);
		System.out.println("outputValue: " + this.outputValue);
		System.out.println("biasOutputWeight: " + this.biasOutputWeight);
		
	}

	private double sigmoid(double net) {
		return (1/( 1 + Math.pow(Math.E,(-1*net))));
	}
}
