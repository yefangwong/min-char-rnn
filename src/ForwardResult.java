public class ForwardResult {
    public int T; // 時間步數
    public int H; // 隱藏層大小
    public int V; // 詞彙表大小

    public double[][] h;      // Hidden states [T][H]
    public double[][] hPrev;  // Previous hidden states [T][H]
    public double[][] c;      // Cell states [T][H]
    public double[][] cPrev;  // Previous cell states [T][H]
    
    // 四大閘門的中間狀態
    public double[][] is;     // Input gates [T][H]
    public double[][] fs;     // Forget gates [T][H]
    public double[][] os;     // Output gates [T][H]
    public double[][] gs;     // Candidate cell states [T][H]
    
    public double[][] y;      // Output probabilities [T][V]
    public double[][] z;      // Output logits [T][V]

    public ForwardResult() {}

    public ForwardResult(int T, int H, int V) {
        this.T = T;
        this.H = H;
        this.V = V;
        this.h = new double[T][H];
        this.hPrev = new double[T][H];
        this.c = new double[T][H];
        this.cPrev = new double[T][H];
        this.is = new double[T][H];
        this.fs = new double[T][H];
        this.os = new double[T][H];
        this.gs = new double[T][H];
        this.y = new double[T][V];
        this.z = new double[T][V];
    }
}
