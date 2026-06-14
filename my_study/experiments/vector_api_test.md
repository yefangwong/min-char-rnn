# Exp-005: Vector API (SIMD) 效能驗證
- **背景:** 既然佈局差異被 JIT 抵消，那麼真正的突破可能在於硬體向量化。
- **對象:** `Standard Loop` vs. `Java Vector API`
- **指標:** GFLOPS, CPU 指令集調用率 (使用 hsdis 查看)
