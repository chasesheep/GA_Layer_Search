# 🚀 快速开始 - GA层替换搜索

## 📋 开始前检查清单

- [ ] 激活conda环境：`conda activate ga_layer_search`
- [ ] 检查GPU空闲：`nvidia-smi`
- [ ] 进入目录：`cd /home/huzhuangfei/Code/GandA/genetic_layer_search`
- [ ] 创建screen会话：`screen -S ga_search`

---

## ⚡ 方式1：快速测试（强烈推荐首次运行）

```bash
cd /home/huzhuangfei/Code/GandA/genetic_layer_search
./quick_test_real.sh
```

**配置**：
- 🎮 GPU: 自动选择空闲GPU
- 📊 种群: 20，代数: 15，无改进: 8  
- ⚡ Limit: 10(快速) / 50(中等)
- ⏱️  预计时间: **4-5小时**

**输出**：
- 结果：`results/real_test/search_result_*.json`
- 日志：`results/real_test/search_log_*.txt`
- 检查点：`results/real_test/checkpoints_*/`

---

## 🎯 方式2：完整生产搜索

```bash
cd /home/huzhuangfei/Code/GandA/genetic_layer_search

# 使用screen后台运行（推荐）
screen -S ga_full
./run_full_search.sh
# 按Ctrl+A, 然后按D退出screen（程序继续运行）

# 重新连接
screen -r ga_full
```

**配置**：
- 🎮 GPU: 3
- 📊 种群: 40，代数: 20，无改进: 6
- ⚡ Limit: 50(快速) / None(完整MMLU)
- ⏱️  预计时间: **~7天**（实际可能3-5天）

**输出**：
- 结果：`results/real_results/search_result_*.json`
- 日志：`results/real_results/search_log_*.txt`
- 检查点：`results/real_results/checkpoints_*/`

---

## 📊 监控进度

### 方式1：查看检查点（推荐）

```bash
# 快速查看进度
./monitor_progress.sh results/real_test

# 或
./monitor_progress.sh results/real_results
```

**每3代自动保存检查点，包含**：
- ✅ 当前代数和评估次数
- ✅ 全局最优解和各层数最优  
- ✅ 种群Top-10
- ✅ 发现的模式（1/2/3层）
- ✅ 统计信息

### 方式2：实时查看日志

```bash
tail -f results/real_test/search_log_*.txt
```

### 方式3：检查GPU

```bash
nvidia-smi
# 或持续监控
watch -n 5 nvidia-smi
```

---

## ⏰ 预期时间线

### 快速测试（4-5小时）
```
00:00 - 启动
00:30 - 第1次检查（应有2-3个检查点）
02:00 - 中期检查  
04:00 - 阶段1完成
05:00 - 全部完成 ✓
```

### 完整搜索（7天）
```
Day 1-2 - 阶段1: GA粗搜索（每12小时检查）
Day 3   - 阶段2: 完整评估（每6小时检查）
Day 3-7 - 阶段3: 局部优化（每12小时检查）
Day 7   - 完成 ✓

注: 实际可能3-5天完成
```

---

## 🔧 常见问题

### Q1: 如何知道程序还在运行？

```bash
# 检查进程
ps aux | grep run_ga

# 查看最新检查点
./monitor_progress.sh results/real_test

# 查看日志最后几行
tail -20 results/real_test/search_log_*.txt
```

### Q2: 程序意外终止怎么办？

1. 查看最新检查点确认进度
   ```bash
   ./monitor_progress.sh results/real_test
   ```

2. 查看日志找错误
   ```bash
   tail -100 results/real_test/search_log_*.txt
   ```

3. 重新运行（修改output-dir避免覆盖）
   ```bash
   python run_ga_search_real.py --output-dir results/real_test_retry
   ```

### Q3: 如何暂停/恢复？

**暂停**：
- 如在screen中：`Ctrl+C` 终止
- 如在后台：`kill <PID>`

**恢复**：
- 目前不支持断点续传
- 需要重新开始（但可以查看检查点了解之前进度）

### Q4: GPU内存不足？

```bash
# 换个GPU
python run_ga_search_real.py --gpu 4

# 或等待当前GPU空闲
nvidia-smi
```

---

## 📈 查看结果

### 最终结果

```bash
# 查看JSON结果
cat results/real_test/search_result_*.json | jq '.final_results'

# 查看发现的模式
cat results/real_test/search_result_*.json | jq '.discovered_patterns'

# 查看统计
cat results/real_test/search_result_*.json | jq '.statistics'
```

### 检查点历史

```bash
# 列出所有检查点
python view_checkpoint.py results/real_test/checkpoints_*/ --list

# 查看特定检查点
python view_checkpoint.py results/real_test/checkpoints_*/checkpoint_gen006.json
```

---

## 📚 更多文档

- 📖 **README.md** - 完整使用指南
- 🏗️ **ARCHITECTURE.md** - 系统架构
- 📋 **USAGE_GUIDE.md** - 详细使用说明
- 📊 **SUMMARY.md** - 项目总结

---

## ✨ 一句话启动

```bash
# 快速测试（4-5小时）
cd /home/huzhuangfei/Code/GandA/genetic_layer_search && screen -S ga_test && ./quick_test_real.sh

# 完整搜索（~7天）
cd /home/huzhuangfei/Code/GandA/genetic_layer_search && screen -S ga_full && ./run_full_search.sh
```

---

**准备好了？开始你的搜索之旅！** 🎉

建议：先运行**快速测试**验证一切正常，再启动完整搜索。

