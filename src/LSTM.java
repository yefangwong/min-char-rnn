/**
 * Minimal character-level LSTM model. Written by Mark Wong (@yefangwong) BSD License
 *
 * Copyright (c) 2026, Mark Wong
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided
 * that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
 * following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
 * the following disclaimer in the documentation and/or other materials provided with the distribution.
 * 3. Neither the name of Mark Wong nor the names of its contributors may be used to endorse or promote products
 * derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * ---
 *
 * This software was developed based on code referenced from karpathy/char-rnn, available at
 * https://github.com/karpathy/char-rnn. The original code is licensed under the
 * BSD license.
 */

import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class LSTM {
    private static final int HIDDEN_SIZE = 100; // 隱藏層大小
    private static final int SEQ_LENGTH = 25;   // 序列長度
    private static final double LEARNING_RATE = 0.1; 
    private static int iterations = 31200;

    // LSTM 權重：wxh 與 whh 包含了 4 個門 (i, f, o, g)
    // 矩陣大小為 [4 * HIDDEN_SIZE][...]
    private double[][] wxh; // 輸入層到隱藏層
    private double[][] whh; // 隱藏層到隱藏層
    private double[] bh;    // 隱藏層偏差
    
    private double[][] why; // 輸出層權重 [vocabSize][HIDDEN_SIZE]
    private double[] by;    // 輸出層偏差 [vocabSize]

    // Adagrad 記憶變數 (優化器用)
    private double[][] mWxh, mWhh, mWhy;
    private double[] mBh, mBy;
    private static final double EPSILON = 1e-8;

    private int vocabSize;
    private Map<Character, Integer> charToIdx;
    private Map<Integer, Character> idxToChar;

    public LSTM(String data) {
        if (data.isEmpty()) return;

        char[] chars = data.toCharArray();
        charToIdx = new HashMap<>();
        idxToChar = new HashMap<>();
        int index = 0;
        for (char c : chars) {
            if (!charToIdx.containsKey(c)) {
                charToIdx.put(c, index);
                idxToChar.put(index, c);
                index++;
            }
        }
        vocabSize = charToIdx.size();

        // 初始化權重：4 * HIDDEN_SIZE 包含 i, f, o, g 四個門控
        wxh = randomMatrix(4 * HIDDEN_SIZE, vocabSize);
        whh = randomMatrix(4 * HIDDEN_SIZE, HIDDEN_SIZE);
        bh = new double[4 * HIDDEN_SIZE];
        
        why = randomMatrix(vocabSize, HIDDEN_SIZE);
        by = new double[vocabSize];
        
        // 初始化 Adagrad 記憶變數
        mWxh = new double[4 * HIDDEN_SIZE][vocabSize];
        mWhh = new double[4 * HIDDEN_SIZE][HIDDEN_SIZE];
        mWhy = new double[vocabSize][HIDDEN_SIZE];
        mBh = new double[4 * HIDDEN_SIZE];
        mBy = new double[vocabSize];
    }

    private double[][] randomMatrix(int rows, int cols) {
        Random rand = new Random();
        double scale = Math.sqrt(6.0 / (rows + cols));
        double[][] matrix = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = rand.nextGaussian() * scale;
            }
        }
        return matrix;
    }

    // 前向傳播 (Forward Pass)
    private void forward(int[] inputs, double[] hPrev, double[] cPrev) {
        int T = inputs.length;
        int H = HIDDEN_SIZE;

        // 宣告隱藏狀態與細胞狀態的暫存空間
        double[][] hs = new double[T][H];
        double[][] cs = new double[T][H];
        
        // LSTM 的門控狀態 (為了反向傳播，之後需要存進 ForwardResult)
        double[][] is = new double[T][H]; // Input gates
        double[][] fs = new double[T][H]; // Forget gates
        double[][] os = new double[T][H]; // Output gates
        double[][] gs = new double[T][H]; // Candidate cell gates (g)

        // 您要求的：在迴圈外宣告當前的長期與短期記憶軌道
        double[] h_t_minus_1 = hPrev; // 短期記憶 (Hidden State)
        double[] c_t_minus_1 = cPrev; // 長期記憶 (Cell State)

        for (int t = 0; t < T; t++) {
            // 這裡將開始實作遺忘與更新邏輯...
        }
    }

    // 數學輔助函數
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double[] matrixVectorMultiply(double[][] matrix, double[] vector) {
        double[] result = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < vector.length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return result;
    }

    private double[] add(double[] a, double[] b) {
        double[] res = new double[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] + b[i];
        return res;
    }
}
