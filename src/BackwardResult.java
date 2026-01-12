public class BackwardResult {
    public double[][] dwxh;
    public double[][] dwhh;
    public double[][] dwhy;
    public double[] dbh;
    public double[] dby;
    
    // 計算所有梯度的L2範數，用於檢測梯度爆炸
    public double calculateGradientNorm() {
        double sumSquares = 0.0;
        
        // 計算 dwxh 的平方和
        for (int i = 0; i < dwxh.length; i++) {
            for (int j = 0; j < dwxh[0].length; j++) {
                sumSquares += dwxh[i][j] * dwxh[i][j];
            }
        }
        
        // 計算 dwhh 的平方和
        for (int i = 0; i < dwhh.length; i++) {
            for (int j = 0; j < dwhh[0].length; j++) {
                sumSquares += dwhh[i][j] * dwhh[i][j];
            }
        }
        
        // 計算 dwhy 的平方和
        for (int i = 0; i < dwhy.length; i++) {
            for (int j = 0; j < dwhy[0].length; j++) {
                sumSquares += dwhy[i][j] * dwhy[i][j];
            }
        }
        
        // 計算 dbh 的平方和
        for (int i = 0; i < dbh.length; i++) {
            sumSquares += dbh[i] * dbh[i];
        }
        
        // 計算 dby 的平方和
        for (int i = 0; i < dby.length; i++) {
            sumSquares += dby[i] * dby[i];
        }
        
        // 返回L2範數（平方和的平方根）
        return Math.sqrt(sumSquares);
    }

    public void scaleGradients(double scale) {
        // 缩放权重梯度
        scaleMatrix(dwxh, scale);
        scaleMatrix(dwhh, scale);
        scaleMatrix(dwhy, scale);

        // 缩放偏置梯度
        scaleArray(dbh, scale);
        scaleArray(dby, scale);
    }

    private void scaleMatrix(double[][] matrix, double scale) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] *= scale;
            }
        }
    }

    private void scaleArray(double[] array, double scale) {
        for (int i = 0; i < array.length; i++) {
            array[i] *= scale;
        }
    }
}
