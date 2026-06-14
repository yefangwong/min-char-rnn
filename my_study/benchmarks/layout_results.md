# Exp-001 Benchmark Results (2026-04-21)
- **環境:** Java 21, 8000x8000 Matrix-Vector Multiply
- **Test A (double[][]):** 82ms (Verified)
- **Test B (double[]):** 84ms (Verified)
- **結論:** 兩者無顯著差異，JIT 成功抵消了指標跳轉開銷。
