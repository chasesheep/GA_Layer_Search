# GA搜索使用指南

## 🚀 开始搜索

### 方式1：快速测试（推荐首次运行）

```bash
cd /home/huzhuangfei/Code/GandA/genetic_layer_search

# 直接运行快速测试脚本
./quick_test_real.sh

# 或使用screen后台运行
screen -S ga_test
./quick_test_real.sh
# Ctrl+A, D 退出screen
```

**配置**：
- 种群: 20
- 代数: 15
- 无改进阈值: 8
- limit: 10/50
- 预计时间: 4-5小时

### 方式2：完整生产搜索

```bash
cd /home/huzhuangfei/Code/GandA/genetic_layer_search

# 使用screen后台运行
screen -S ga_full
./run_full_search.sh
# Ctrl+A, D 退出screen
```

**配置**：
- 种群: 40
- 代数: 20
- 无改进阈值: 6
- limit: 50/None
- 预计时间: ~7天（实际可能3-5天）

### 方式3：自定义参数

```bash
python run_ga_search_real.py \
  --gpu 3 \
  --fast-limit 50 \
  --full-limit None \
  --population 40 \
  --generations 20 \
  --no-improve 6 \
  --output-dir results/real_results \
  --verbose
```

---

## 📊 监控进度

### 1. 查看检查点（每3代自动保存）

```bash
# 列出所有检查点
python view_checkpoint.py results/real_results/checkpoints_YYYYMMDD_HHMMSS/ --list

# 查看最新检查点
python view_checkpoint.py results/real_results/checkpoints_YYYYMMDD_HHMMSS/

# 或使用监控脚本
./monitor_progress.sh results/real_results
```

**检查点包含**：
- ✅ 当前代数和评估次数
- ✅ 全局最优解
- ✅ 各层数最优解（2/3/4层）
- ✅ 种群Top-10
- ✅ 发现的模式（1/2/3层，带频率和质量分数）
- ✅ 统计信息（层覆盖、无改进次数）

### 2. 实时查看日志

```bash
# 实时跟踪日志
tail -f results/real_results/search_log_YYYYMMDD_HHMMSS.txt

# 查看最近100行
tail -100 results/real_results/search_log_YYYYMMDD_HHMMSS.txt

# 搜索关键信息
grep "代.*最优" results/real_results/search_log_YYYYMMDD_HHMMSS.txt
grep "检查点" results/real_results/search_log_YYYYMMDD_HHMMSS.txt
```

### 3. 检查GPU状态

```bash
# 查看GPU使用
nvidia-smi

# 监控GPU
watch -n 5 nvidia-smi
```

### 4. 检查运行状态

```bash
# 查看进程
ps aux | grep run_ga_search

# 重新连接screen
screen -r ga_test   # 或 ga_full
```

---

## 🔧 故障处理

### 情况1：进程意外终止

**症状**：检查点保存了，但进程不在运行

**恢复**：
1. 查看最新检查点确认进度
   ```bash
   ./monitor_progress.sh results/real_results
   ```

2. 查看日志找到错误原因
   ```bash
   tail -100 results/real_results/search_log_*.txt
   ```

3. 如果是可恢复错误，重新运行（注意修改output-dir避免覆盖）
   ```bash
   python run_ga_search_real.py --output-dir results/real_results_retry
   ```

### 情况2：搜索卡住

**症状**：很长时间没有新的检查点

**检查**：
1. 查看GPU使用
   ```bash
   nvidia-smi
   ```

2. 查看最新日志
   ```bash
   tail -f results/real_results/search_log_*.txt
   ```

3. 如果MMLU评估卡住（常见），可能需要重启

### 情况3：GPU内存不足

**症状**：OOM错误

**解决**：
1. 检查其他进程
   ```bash
   nvidia-smi
   ```

2. 使用不同GPU
   ```bash
   python run_ga_search_real.py --gpu 4
   ```

3. 或等待GPU空闲

---

## 📈 查看结果

### 最终结果文件

```bash
# JSON结果
cat results/real_results/search_result_YYYYMMDD_HHMMSS.json | jq

# 查看最优解
cat results/real_results/search_result_YYYYMMDD_HHMMSS.json | jq '.final_results'

# 查看模式
cat results/real_results/search_result_YYYYMMDD_HHMMSS.json | jq '.discovered_patterns'
```

### 分析脚本（TODO）

```bash
# 统计分析
python analyze_results.py results/real_results/search_result_*.json

# 可视化
python plot_results.py results/real_results/search_result_*.json
```

---

## 💡 最佳实践

### 1. 运行前检查

- [ ] GPU是否空闲 (`nvidia-smi`)
- [ ] 环境已激活 (`which python`)
- [ ] 磁盘空间充足 (`df -h`)
- [ ] screen/tmux会话已创建

### 2. 运行中监控

- [ ] 定期查看检查点（每小时）
  ```bash
  ./monitor_progress.sh results/real_results
  ```

- [ ] 检查日志是否有错误
  ```bash
  tail -100 results/real_results/search_log_*.txt | grep -i error
  ```

- [ ] 验证进度是否正常（适应度是否提升）

### 3. 运行后分析

- [ ] 检查最终结果
  ```bash
  python view_checkpoint.py results/real_results/checkpoints_*/
  ```

- [ ] 对比不同运行的结果
- [ ] 分析发现的模式
- [ ] 准备论文素材

---

## 📋 常用命令速查

```bash
# === 启动搜索 ===
./quick_test_real.sh                          # 快速测试
./run_full_search.sh                          # 完整搜索
screen -S ga && ./run_full_search.sh          # 后台运行

# === 监控进度 ===
./monitor_progress.sh results/real_results    # 查看检查点
tail -f results/real_results/search_log_*.txt # 实时日志
screen -r ga                                   # 重新连接

# === 查看结果 ===
python view_checkpoint.py results/real_results/checkpoints_*/ --list
ls -lth results/real_results/                  # 列出所有文件
cat results/real_results/search_result_*.json | jq '.final_results'

# === GPU监控 ===
nvidia-smi                                     # 当前状态
watch -n 5 nvidia-smi                         # 持续监控

# === 故障处理 ===
ps aux | grep run_ga                          # 检查进程
tail -100 results/real_results/search_log_*.txt  # 查看错误
```

---

## ⏰ 时间规划

### 快速测试（4-5小时）
```
00:00 - 启动运行
00:30 - 第一次检查（应该有2-3个检查点）
01:00 - 第二次检查
02:00 - 中期检查
04:00 - 完成阶段1
05:00 - 全部完成
```

### 完整搜索（~7天，可能更快）
```
Day 1-2 - 阶段1 (GA粗搜索, ~56小时)
  每12小时检查一次进度
  
Day 3 - 阶段2 (完整评估, ~11小时)
  每6小时检查一次
  
Day 3-7 - 阶段3 (局部优化, ~100小时)
  每12小时检查一次
  
Day 7 - 完成和分析

注: 由于缓存和优化，实际可能只需3-5天
```

---

**准备好了吗？让我们开始搜索！** 🚀

```bash
cd /home/huzhuangfei/Code/GandA/genetic_layer_search
./quick_test_real.sh
```

