# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tester 
@File    ：demo.py
@Author  ：Huang Zheng
@Date    ：2025/7/12 20:46
'''

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from gemini_webapi import GeminiClient  # 修正：使用 GeminiClient
from gemini_webapi.constants import Model
import os
import json
import time
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Annotated
from contextlib import asynccontextmanager
import markdown


load_dotenv()

# 从环境变量获取配置
SECURE_1PSID = os.getenv('SECURE_1PSID')
SECURE_1PSIDTS = os.getenv('SECURE_1PSIDTS')
API_KEY = os.getenv('API_KEY')

# 检查必要的环境变量
if not API_KEY:
    raise ValueError("API_KEY environment variable is required")
if not SECURE_1PSID:
    raise ValueError("SECURE_1PSID environment variable is required")

# 全局 Gemini 客户端
gemini_client = None


async def get_gemini_client():
    """获取或初始化 Gemini 客户端"""
    global gemini_client
    if gemini_client is None:
        # 修正：使用正确的参数名（小写）
        gemini_client = GeminiClient(
            secure_1psid=SECURE_1PSID,
            secure_1psidts=SECURE_1PSIDTS or "",  # 如果没有可以为空
            proxy=None
        )
        await gemini_client.init(timeout=30, auto_close=True, auto_refresh=True, verbose=True)
    return gemini_client


# 修正：使用 lifespan 替代 on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时的初始化
    yield
    # 关闭时的清理
    global gemini_client
    if gemini_client:
        await gemini_client.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 验证 API Key
def verify_api_key_header(authorization: Annotated[str | None, Header()] = None):
    """验证 Header 中的 API Key"""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 支持 "Bearer xxx" 格式
    if authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
    else:
        # 支持直接传入 token
        token = authorization

    if token != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


# OpenAI API 兼容的数据模型
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gemini-2.5-pro"
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None


@app.api_route("/", methods=['GET','HEAD'])
async def root():
    return {"message": "Gemini to OpenAI API Bridge", "status": "running"}


@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key_header)):
    """列出可用模型 - OpenAI API 兼容"""
    return {
        "object": "list",
        "data": [
            {
                "id": "unspecified",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google"
            },
            {
                "id": "gemini-2.5-flash",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google"
            },
            {
                "id": "gemini-2.5-pro",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google"
            }
        ]
    }


def get_model_name(model: str) -> str:
    """
    映射模型名称到 gemini_webapi 支持的模型
    """
    # 修正：根据实际的模型映射进行调整
    model_mapping = {
        # "gemini-pro": "G_2_5_PRO",  # 另一个可能的默认模型
        # "gemini-2.0-flash": "G_2_0_FLASH",
        # "gemini-2.5-flash": "G_2_5_FLASH"
        'gemini-2.5-flash': Model.G_2_5_FLASH,
        'gemini-2.5-pro': Model.G_2_5_PRO,

    }
    return model_mapping.get(model, Model.UNSPECIFIED)


@app.post("/v1/chat/completions")
async def chat_completions(
        request: ChatCompletionRequest,
        api_key: str = Depends(verify_api_key_header)
):
    """ChatGPT API 兼容的聊天完成端点"""
    try:
        # 获取 Gemini 客户端
        client = await get_gemini_client()

        # 提取最后一条用户消息
        user_message = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break

        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        # 选择模型
        model_name = get_model_name(request.model)

        # 生成响应
        response = await client.generate_content(user_message, model=model_name)
        response_text = response.text
        
        if response_text:
            #response_text = response_text.replace('\\n', '\n')
            response_text = response_text.replace('\\n', r'<br>')

        #debug
        request.stream = False
        
        if request.stream:
            # 修正：使用正确的 content-type
            return StreamingResponse(
                generate_stream_response(response_text, request.model),
                media_type="text/event-stream"
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


# 健康检查端点（不需要认证）
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
