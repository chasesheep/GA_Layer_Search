#!/bin/bash
# 检查GA搜索的tmux会话状态

echo "========================================================================"
echo "GA搜索 - Tmux会话状态"
echo "========================================================================"
echo ""

# 列出所有tmux会话
echo "📊 运行中的tmux会话:"
tmux ls 2>/dev/null || echo "没有运行的tmux会话"
echo ""

# 检查ga_quick
if tmux has-session -t ga_quick 2>/dev/null; then
    echo "=== ga_quick (快速测试: GPU 3, limit=10/50) ==="
    tmux capture-pane -t ga_quick -p | tail -15
    echo ""
fi

# 检查ga_full_fast  
if tmux has-session -t ga_full_fast 2>/dev/null; then
    echo "=== ga_full_fast (完整搜索: GPU 4, limit=20/None) ==="
    tmux capture-pane -t ga_full_fast -p | tail -15
    echo ""
fi

echo "========================================================================"
echo "提示:"
echo "  - 连接会话: tmux attach -t ga_quick  或  tmux attach -t ga_full_fast"
echo "  - 退出会话: Ctrl+B, 然后按 D"
echo "  - 查看检查点: ./monitor_progress.sh results/real_test"
echo "  - 查看GPU: nvidia-smi"
echo "========================================================================"

