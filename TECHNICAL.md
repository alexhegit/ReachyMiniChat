# ReachyClaw 技术方案

> 项目目标：由 OpenClaw 驱动的 Reachy Mini 虚拟数字人

## 1. 项目背景

参考项目：
- [ReachyMiniChat](https://github.com/alexhegit/ReachyMiniChat) - 基于 Ollama 的实现
- [Reachy Mini Simulation](https://huggingface.co/docs/reachy_mini/platforms/simulation/get_started) - 模拟器文档
- [OpenClaw](https://github.com/openclaw/openclaw) - AI Agent 框架

## 2. 系统架构

```
┌─────────────┐     WebSocket      ┌──────────────┐
│  用户 (TUI) │ ◄────────────────► │  ReachyClaw  │
└─────────────┘                    │   (Python)   │
                                    └──────┬───────┘
                                           │ HTTP/WS
                                    ┌──────▼───────┐
                                    │   OpenClaw   │
                                    │   (大脑)     │
                                    └──────────────┘
```

## 3. 技术选型

| 模块 | 技术选型 | 说明 |
|------|----------|------|
| **TUI** | Python (rich/textual) | 文本+语音交互界面 |
| **ASR** | faster-whisper | 语音识别，复用 ReachyMiniChat |
| **LLM** | OpenClaw | 大脑，可切换多模型 |
| **TTS** | edgeTTS / espeak | 语音合成，复用 ReachyMiniChat |
| **机器人** | reachy-mini SDK | 连接模拟器 localhost |

## 4. OpenClaw 集成

### 4.1 启用 HTTP API

在 `~/.openclaw/openclaw.json` 中配置：

```json
{
  "gateway": {
    "http": {
      "endpoints": {
        "chatCompletions": {
          "enabled": true
        }
      }
    }
  }
}
```

### 4.2 API 调用示例

```bash
curl -X POST http://127.0.0.1:18789/v1/chat/completions \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openclaw",
    "messages": [{"role":"user","content":"hello"}]
  }'
```

### 4.3 模型配置

当前配置的模型：
- 默认：`openclaw` (OpenClaw 代理)
- 说明：OpenClaw 作为 AI Agent 框架，统一管理后端大模型

## 5. 实现路径

### 路径 A：复用 ReachyMiniChat（推荐）

将 `ReachyMiniChat/emo_v8.py` 中的 Ollama 调用替换为 OpenClaw：

**已创建文件**：
- `ReachyClaw/emo_v8_openclaw.py` - 基于 emo_v8，集成 OpenClaw

**改动点**：
1. `_get_ollama_response_async()` → `_get_openclaw_response_async()`
2. 使用 OpenAI 兼容 API (`/v1/chat/completions`)
3. 支持流式输出

### 路径 B：重新设计

利用 OpenClaw 的 tool calling 能力，让模型直接返回结构化动作指令。

## 6. 运行方式

### 6.1 启动 Reachy Mini 模拟器

```bash
reachy-mini-daemon --sim
```

### 6.2 启动 OpenClaw Gateway

确保 HTTP API 已启用：
```bash
openclaw gateway restart
```

### 6.3 运行 Chat

```bash
# 文本聊天模式
cd ReachyClaw
python emo_v8_openclaw.py

# ASR 语音模式
python emo_v8_openclaw.py --asr
```

## 7. 下一步

- [ ] 测试 emo_v8_openclaw.py 连通性
- [ ] 配置更多模型
- [ ] 添加动作指令解析（让 LLM 返回结构化动作）
- [ ] 优化延迟

## 8. 参考资料

- [OpenClaw OpenAI HTTP API 文档](https://docs.openclaw.ai/gateway/openai-http-api)
- [Reachy Mini Simulation 文档](https://huggingface.co/docs/reachy_mini/platforms/simulation/get_started)
- [ReachyMiniChat 项目](https://github.com/alexhegit/ReachyMiniChat)
