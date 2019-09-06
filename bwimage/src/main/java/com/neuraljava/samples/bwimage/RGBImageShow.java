package com.neuraljava.samples.bwimage;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;

public class RGBImageShow {

	/**
	 * Mostra a composição de uma imagem monocromática
	 * @param args String - Path da imagem
	 * @throws IOException 
	 */
	public static void main(String [] args) throws IOException {
		BufferedImage image = ImageIO.read(new File(args[0]));
		drawImage(image);
		Color[][] pixels = new Color[image.getWidth()][];

		for (int x = 0; x < image.getWidth(); x++) {
		    pixels[x] = new Color[image.getHeight()];

		    for (int y = 0; y < image.getHeight(); y++) {
		        pixels[x][y] = new Color(image.getRGB(y,x));
		    }
		}
		for (int x=0; x < pixels.length; x++) {
			for (int y = 0; y < pixels[x].length; y++) { 
				System.out.print(
						"["
						+ String.format("%3d", pixels[x][y].getRed())
						+ "-" 
						+ String.format("%3d", pixels[x][y].getGreen())
						+ "-"
						+ String.format("%3d", pixels[x][y].getBlue())
						+ "]"
				);
			}
			System.out.println("");
		}
	}
	

	private static void drawImage(BufferedImage image) {
		JLabel picLabel = new JLabel(new ImageIcon(image));
		JOptionPane.showMessageDialog(null, picLabel, "Imagem", JOptionPane.PLAIN_MESSAGE, null);
	}
	
}
