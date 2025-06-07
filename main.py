from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from gemini_webapi import Gemini
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 从环境变量获取 Cookie
SECURE_1PSID = os.getenv('SECURE_1PSID')
SECURE_1PSIDTS = os.getenv('SECURE_1PSIDTS')

# OpenAI API 兼容的数据模型
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gemini-pro"
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

@app.get("/")
async def root():
    return {"message": "Gemini to OpenAI API Bridge"}

@app.get("/v1/models")
async def list_models():
    """列出可用模型 - OpenAI API 兼容"""
    return {
        "object": "list",
        "data": [
            {
                "id": "gemini-pro",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """ChatGPT API 兼容的聊天完成端点"""
    try:
        # 初始化 Gemini
        gemini = Gemini(
            secure_1psid=SECURE_1PSID,
            secure_1psidts=SECURE_1PSIDTS
        )
        
        # 提取最后一条用户消息
        user_message = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # 生成响应
        response_text = await gemini.generate_content(user_message)
        
        if request.stream:
            # 流式响应
            return StreamingResponse(
                generate_stream_response(response_text, request.model),
                media_type="text/plain"
            )
        else:
            # 非流式响应
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(user_message.split()) + len(response_text.split())
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def generate_stream_response(content: str, model: str):
    """生成流式响应"""
    chunk_id = f"chatcmpl-{int(time.time())}"
    
    # 分块发送内容
    words = content.split()
    for i, word in enumerate(words):
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": word + " " if i < len(words) - 1 else word
                    },
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    
    # 发送结束标记
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
