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
package com.neuraljava.samples.mlpgen.api;

import java.util.List;

/**
 * Representa um nó da rede
 * @author Cleuton Sampaio
 *
 */
public class Node {
	public int layerNumber;
	public int nodeNumber;
	public List<Sinapse> sinapses;
	public double input;  // net value before activation
	public double value;  // output or current value
	@Override
	public boolean equals(Object obj) {
		Node other = (Node) obj;
		return other.input == this.input && other.layerNumber == this.layerNumber;
	}
	@Override
	public String toString() {
		String saida = "\n\t[Node. Layer: " 
					 + this.layerNumber
					 + ", node: "
					 + this.nodeNumber
					 + ", input: "
					 + this.input
					 + ", value: "
					 + this.value
					 + " sinapses: " 
					 + sinapses
					 + "]";
		return saida;
	}
	
}
