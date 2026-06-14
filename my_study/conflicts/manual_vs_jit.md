# manual_flattening_vs_jit_pointer_lifting
**id:** conflict-manual-vs-jit-005
- **假設 A:** 手動將數據攤平為一維陣列是確保效能的唯一途徑。
- **假設 B:** JIT 的 Row Pointer Caching 與 BCE 優化使得雙層陣列在特定規模下效能與一維陣列持平，且更易於維護。
- **衝突點:** 手動索引計算產生的運算負擔 vs. 指標跳轉產生的存取負擔。
