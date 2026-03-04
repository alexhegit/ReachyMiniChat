# EMO_V8_OPENCLAW — OpenClaw API Integration

## Overview

`EMO_V8_OPENCLAW` represents a significant evolution in the Reachy Mini chat system, replacing the Ollama backend with OpenClaw API integration. This implementation maintains all existing features (ASR, TTS, emotion analysis, robot control) while providing seamless integration with OpenClaw's AI agent framework.

## Architecture

```
┌─────────────┐     WebSocket      ┌──────────────┐
│  User (TUI) │ ◄────────────────► │  ReachyClaw  │
└─────────────┘                    │   (Python)   │
                                   └──────┬───────┘
                                          │ HTTP/WS
                                   ┌──────▼───────┐
                                   │   OpenClaw   │
                                   │   (Brain)    │
                                   └──────────────┘
```

## Key Features

- **OpenClaw API Integration**: Uses OpenAI-compatible HTTP API endpoints
- **Proper Authentication**: Implements Bearer token authentication with required headers
- **Maintains All Existing Features**: ASR (faster-whisper), TTS (Edge-TTS), emotion analysis, and robot control
- **Streaming Support**: Real-time response streaming with parallel actions
- **ASR Integration**: Voice activity detection (VAD) for hands-free interaction

## Implementation Details

### API Configuration

The OpenClaw API expects:
- Model name: `openclaw` (not the original model name)
- Authorization header: `Authorization: Bearer <token>`
- Custom header: `x-openclaw-agent-id: main`
- Endpoint: `/v1/chat/completions`

### Authentication Headers

```python
headers = {
    "Authorization": f"Bearer {self.openclaw_token}",
    "Content-Type": "application/json",
    "x-openclaw-agent-id": "main"
}
```

### Payload Structure

```python
payload = {
    "model": "openclaw",  # OpenClaw-specific model name
    "messages": [
        {"role": "system", "content": "You are a cute desktop robot assistant..."},
        {"role": "user", "content": prompt}
    ],
    "max_tokens": 200,
    "temperature": 0.8,
    "stream": True  # For streaming responses
}
```

## Setup Instructions

### 1. Configure OpenClaw Gateway

In `~/.openclaw/openclaw.json`, enable the HTTP API. For detailed configuration options, see the official OpenClaw documentation: [docs/gateway/openai-http-api.md](https://github.com/openclaw/openclaw/blob/5d51e995/docs/gateway/openai-http-api.md#L12-L12)

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

### 2. Find Your OpenClaw Token

Your OpenClaw token can be found in the `~/.openclaw/openclaw.json` configuration file. Look for the `auth` section:

```json
{
  "auth": {
    "gatewayToken": "YOUR_GATEWAY_TOKEN_HERE"
  }
}
```

Alternatively, you can retrieve it using the OpenClaw CLI:

```bash
openclaw config get auth.gatewayToken
```

### 3. Restart OpenClaw Gateway

```bash
openclaw gateway restart
```

### 4. Verify OpenClaw API

Test the API with curl (replace YOUR_TOKEN with your actual token):

```bash
curl -sS http://127.0.0.1:18789/v1/chat/completions \
  -H 'Authorization: Bearer YOUR_TOKEN' \
  -H 'Content-Type: application/json' \
  -H 'x-openclaw-agent-id: main' \
  -d '{
    "model": "openclaw",
    "messages": [{"role":"user","content":"hi"}]
  }'
```

### 5. Run the Application

```bash
# Text chat mode
python emo_v8_openclaw.py --token "your_openclaw_token"

# ASR voice mode
python emo_v8_openclaw.py --token "your_openclaw_token" --asr
```

## Usage Examples

### Basic Text Chat
```bash
python emo_v8_openclaw.py --token "your_openclaw_gateway_token"
```

### ASR Voice Mode
```bash
python emo_v8_openclaw.py --token "your_openclaw_gateway_token" --asr
```

### With Debug Information
```bash
python emo_v8_openclaw.py --token "your_openclaw_gateway_token" --debug
```

## Key Differences from Previous Versions

| Feature | emo_v7 (Ollama) | emo_v8_openclaw (OpenClaw) |
|---------|-----------------|----------------------------|
| Backend | Ollama | OpenClaw API |
| Authentication | None required | Bearer token required |
| Model Name | Specific LLM names | "openclaw" |
| Headers | Standard | Custom headers required |
| API Type | Local OllMA | Remote OpenClaw |

## Troubleshooting

### Common Issues

1. **401 Unauthorized Error**
   - Verify your OpenClaw token is correct
   - Check that the Authorization header is properly formatted

2. **Connection Refused**
   - Ensure OpenClaw Gateway is running
   - Verify the API endpoint URL is correct

3. **Model Not Found**
   - Confirm the model name is "openclaw" (not specific LLM names)

### Verification Steps

1. Test the API directly with curl as shown above
2. Verify OpenClaw Gateway is running: `openclaw gateway status`
3. Check that the chatCompletions endpoint is enabled in the config

## Performance Notes

- **Latency**: Typical response times around 4-6 seconds depending on OpenClaw server performance
- **Streaming**: Real-time streaming provides immediate feedback during response generation
- **ASR Integration**: Voice-to-text conversion happens locally using faster-whisper
- **TTS Quality**: Uses Edge-TTS for high-quality, multi-language voice synthesis

## Future Enhancements

- Direct action command parsing from LLM responses
- Enhanced emotion analysis with OpenClaw's tool calling
- Multi-agent collaboration features
- Advanced voice customization options