#!/usr/bin/env python
# coding: utf-8

"""
vLLM 服务器启动脚本
=================
此脚本通过FastAPI (uvicorn)启动vLLM作为本地Python服务器，
并提供'/generate'的HTTP接口。

使用示例:
    CUDA_VISIBLE_DEVICES=0 python3 minimal_r1/launch_vllm.py --model_name Seungyoun/Qwen2.5-7B-Open-R1-Distill

服务器启动后，访问 http://localhost:8000/docs 可以查看Swagger UI测试界面。
"""

import uvicorn
import argparse
import io
import torch
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List

# vLLM
from vllm import LLM, SamplingParams

# ----------- 参数解析 -----------
parser = argparse.ArgumentParser(description="使用指定模型启动vLLM服务器")
parser.add_argument("--model_name", type=str, required=True, help="要使用的模型路径或名称")
parser.add_argument("--tensor_parallel_size", type=int, default=2, help="用于张量并行的GPU数量")
args = parser.parse_args()


# ----------- FastAPI初始化 -----------
app = FastAPI(title="vLLM Server", version="0.1")


# ----------- vLLM初始化 -----------
llm = LLM(
    model=args.model_name,
    trust_remote_code=True,
    tensor_parallel_size=args.tensor_parallel_size,  # 使用命令行参数
    dtype="bfloat16"
)
# 默认采样参数
default_params = SamplingParams(
    temperature=0.65,
    top_p=0.95,
    max_tokens=1024,
)


class GenerateRequest(BaseModel):
    prompts: List[str]
    num_gen: int = 1
    temperature: float = 0.65
    max_tokens: int = 1024


class GenerateResponse(BaseModel):
    generations: List[List[str]]


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """使用vLLM为每个prompt生成num_gen个样本"""
    sampling_params = SamplingParams(
        temperature=req.temperature,
        top_p=0.95,
        max_tokens=req.max_tokens,
    )

    num_prompts = len(req.prompts)  # 原始prompt数量
    outputs = llm.generate(req.prompts * req.num_gen, sampling_params)  # 确保乘法正确

    # 初始化正确数量的列表
    generations = [[] for _ in range(num_prompts)]

    # 将输出正确分配回对应的prompt组
    for i, output in enumerate(outputs):
        prompt_idx = i % num_prompts  # 确保正确分组
        generations[prompt_idx].append(output.outputs[0].text)  # 添加到对应列表

    return GenerateResponse(generations=generations)



@app.post("/load_weights")
async def load_weights(request: Request):
    """
    将PyTorch的state_dict加载到vLLM模型中。
    客户端可以通过以下方式发送state_dict：

    ```python
    import requests, torch, io

    # state_dict (例如：微调后的权重)
    model_sd = torch.load("fine_tuned.pt")

    buf = io.BytesIO()
    torch.save(model_sd, buf)
    buf.seek(0)
    resp = requests.post("http://localhost:8000/load_weights", data=buf.read())

    print(resp.json())
    ```
    """
    if llm is None:
        raise HTTPException(status_code=400, detail="LLM未初始化.")

    try:
        weights_data = await request.body()
        buffer = io.BytesIO(weights_data)

        state_dict = torch.load(buffer, map_location="cpu")  # change to state_dict

        llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(state_dict.items())
        print("\3[32m新模型权重已加载.\3[0m")

        return {"status": "success", "message": "模型权重已加载."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载权重失败：{str(e)}")


if __name__ == "__main__":
    print("正在启动vLLM API服务器，监听端口:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
