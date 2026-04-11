# 🧠 LSTM 研發日誌 (Research Log)

## 📅 日期：2026-04-10 (Friday)
**進度：** 開始實作 `src/LSTM.java`，並研究 LSTM 的門控（Gate）數學原理。

---

## 🔍 專題研究 1：Sigmoid 函數 (S型曲線)

### 1. 數學定義
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
在 Java 中實作為：`1.0 / (1.0 + Math.exp(-x))`。

### 2. 幾何與數值意義
- **範圍：** 固定在 $[0, 1]$ 之間。
- **中心點：** $x=0 \implies \sigma(x)=0.5$。
- **飽和區：** 當 $x$ 極大或極小時，輸出趨近於 $1$ 或 $0$。

> [!TIP] **在 LSTM 中的隱喻：閘門 (Gate)**
> - **輸出趨近於 0：** 代表「門關上了」，資訊不准通過（遺忘/過濾）。
> - **輸出趨近於 1：** 代表「門全開」，資訊 100% 通過（保留/存入）。
> - **中間值：** 實現了「軟性控制」，允許部分資訊流動。

---

## 🔍 專題研究 2：為什麼 LSTM 權重 (Wxh) 是 4 倍大？

### 1. 核心需求：四個門 (Four Gates)
LSTM 每一時間步需要計算 4 個向量來控制記憶：
1. **$i$ (Input Gate)：** 控制新資訊存入長期記憶。
2. **$f$ (Forget Gate)：** 控制舊資訊從長期記憶中丟棄。
3. **$o$ (Output Gate)：** 控制長期記憶如何輸出到隱藏態。
4. **$g$ (Cell Gate / Candidate)：** 準備好要寫入的「新記憶內容」。

### 2. 為什麼要合併成一個 4 倍大的權重矩陣？
> [!IMPORTANT] **效率與打包運算**
> - **傳統做法：** 執行 4 次矩陣乘法（算 $i, f, o, g$），對 CPU/GPU 負擔較重。
> - **優化做法：** 將這四組權重「垂直堆疊」在一起。
>   - 原本：`Wxi`, `Wxf`, `Wxo`, `Wxg` (各 100 列)
>   - 合併：`Wxh` (總共 400 列)
> - **優點：** 只需做 **一次** 大矩陣乘法，就能算出所有門的原始數值 (Raw values)，再用 Array Slice 切開套用激活函數即可。

---

## 💻 程式碼實作進度：`LSTM.java`

### 1. 權重與維度設計 (Weight Dimensions)
- **4 倍門控打包：** `wxh` 大小為 `[4 * HIDDEN_SIZE][vocabSize]`，一次運算同時得出 $i, f, o, g$ 的原始數值。
- **門控分配：**
  - `[0, HIDDEN_SIZE)`：Input Gate ($i$)
  - `[HIDDEN_SIZE, 2*HIDDEN_SIZE)`：Forget Gate ($f$)
  - `[2*HIDDEN_SIZE, 3*HIDDEN_SIZE)`：Output Gate ($o$)
  - `[3*HIDDEN_SIZE, 4*HIDDEN_SIZE)`：Candidate Cell ($g$)

### 2. 優化器準備 (Optimization)
- **Adagrad 記憶變數：** 為每個 4 倍大的權重矩陣（`mWxh`, `mWhh`, `mBh`）建立對應的平方累積矩陣，確保每個參數都有獨立的學習率縮放。

### 3. 前向傳播雙軌制 (Dual-track Forward Pass)
- **長期與短期記憶：** 在 `forward()` 迴圈外初始化 `h_t_minus_1` (短期) 與 `c_t_minus_1` (長期)。
- **中間狀態儲存：** 建立 `is`, `fs`, `os`, `gs` 以及 `hs`, `cs` 二維陣列，用以儲存每個時間步 $t$ 的門控狀態與記憶內容，為反向傳播 (BPTT) 鋪路。

### 4. 基礎數學工具 (Math Utilities)
- 實作 `sigmoid(double x)` 與其向量版。
- 實作 `matrixVectorMultiply` 以處理核心的矩陣與 One-hot 向量運算。

---

## 💡 心得與下一步
- **心得：** LSTM 的「強大」來自於它能透過門控精準地管理長期記憶 ($c_t$)。
- **下一步：** 
  - 研究 **Tanh 函數** 與 **長期記憶更新公式**：$c_t = f \cdot c_{t-1} + i \cdot g$。
  - 在 `LSTM.java` 的 `forward()` 迴圈中實作這 4 個門的「切分與激活」邏輯。
