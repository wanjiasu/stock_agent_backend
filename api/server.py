import asyncio
from datetime import date
from enum import Enum
from functools import partial
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from cli.models import AnalystType
from web.utils.analysis_runner import (
    format_analysis_results,
    run_stock_analysis,
    validate_analysis_params,
)

# 确保加载环境变量（兼容CLI/Web配置）
load_dotenv(override=False)

app = FastAPI(
    title="TradingAgents API",
    description="通过 FastAPI 暴露 TradingAgents 股票分析能力",
    version="0.1.0",
)


class MarketType(str, Enum):
    """市场类型枚举，保持与项目内部标识一致。"""

    CN = "A股"
    HK = "港股"
    US = "美股"


DEFAULT_PROVIDER_MODELS = {
    "dashscope": "qwen-plus",
    "阿里": "qwen-plus",
    "deepseek": "deepseek-chat",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "google": "gemini-2.0-flash",
    "qianfan": "ernie-3.5-8k",
    "siliconflow": "deepseek-v3",
    "openrouter": "meta-llama/llama-4-scout:free",
    "ollama": "llama3.1",
    "default": "gpt-4o-mini",
}


class ProgressMessage(BaseModel):
    message: str
    step: Optional[int] = None
    total_steps: Optional[int] = None


class AnalysisRequest(BaseModel):
    stock_symbol: str = Field(..., alias="ticker", description="股票代码，例如 AAPL/000001")
    analysis_date: date = Field(default_factory=date.today, description="分析日期，默认今天")
    analysts: List[AnalystType] = Field(..., description="分析师列表，例如 ['market', 'fundamentals']")
    research_depth: int = Field(1, ge=1, le=5, description="研究深度，1-5 等级")
    llm_provider: str = Field("dashscope", description="LLM 提供商标识，例如 dashscope/openai")
    llm_model: Optional[str] = Field(None, description="默认使用提供商推荐模型")
    quick_think_model: Optional[str] = Field(None, description="可选的快速思考模型")
    deep_think_model: Optional[str] = Field(None, description="可选的深度思考模型")
    backend_url: Optional[str] = Field(
        None, description="自定义 LLM 服务端点（覆盖默认端点）"
    )
    market_type: MarketType = Field(MarketType.US, description="股票市场类型")

    class Config:
        populate_by_name = True

    @validator("analysts")
    def _ensure_analysts(cls, value: List[AnalystType]):
        if not value:
            raise ValueError("analysts 列表不能为空")
        return value

    @validator("llm_provider")
    def _normalize_provider(cls, value: str) -> str:
        return value.strip()

    @validator("market_type", pre=True)
    def _normalize_market_type(cls, value):
        if isinstance(value, MarketType):
            return value
        mapping = {
            "cn": MarketType.CN,
            "china": MarketType.CN,
            "a股": MarketType.CN,
            "hk": MarketType.HK,
            "hkg": MarketType.HK,
            "港股": MarketType.HK,
            "us": MarketType.US,
            "usa": MarketType.US,
            "美股": MarketType.US,
        }
        key = str(value).strip().lower()
        if key in mapping:
            return mapping[key]
        return MarketType(value)

    @validator("llm_model", always=True)
    def _default_model(cls, value: Optional[str], values) -> str:
        if value:
            return value
        provider = values.get("llm_provider", "dashscope")
        provider_key = provider.lower()
        for key, model in DEFAULT_PROVIDER_MODELS.items():
            if key != "default" and key in provider_key:
                return model
        return DEFAULT_PROVIDER_MODELS["default"]


class AnalysisResponse(BaseModel):
    success: bool
    results: Optional[dict] = None
    formatted_results: Optional[dict] = None
    progress: List[ProgressMessage] = Field(default_factory=list)
    errors: Optional[List[str]] = None


@app.get("/health")
def health_check():
    """健康检查接口。"""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_stock(request: AnalysisRequest):
    """触发股票分析流程，功能等价于 `python -m cli.main analyze`."""

    analysts = [
        analyst.value if isinstance(analyst, AnalystType) else analyst
        for analyst in request.analysts
    ]
    analysis_date_str = request.analysis_date.strftime("%Y-%m-%d")

    # 预验证参数，避免长时间任务后才发现错误
    is_valid, validation_errors = validate_analysis_params(
        stock_symbol=request.stock_symbol,
        analysis_date=analysis_date_str,
        analysts=analysts,
        research_depth=request.research_depth,
        market_type=request.market_type.value,
    )
    if not is_valid:
        raise HTTPException(status_code=422, detail=validation_errors)

    progress_events: List[ProgressMessage] = []

    def progress_callback(message: str, step: int = None, total_steps: int = None):
        progress_events.append(
            ProgressMessage(message=message, step=step, total_steps=total_steps)
        )

    runner = partial(
        run_stock_analysis,
        stock_symbol=request.stock_symbol,
        analysis_date=analysis_date_str,
        analysts=analysts,
        research_depth=request.research_depth,
        llm_provider=request.llm_provider.lower(),
        llm_model=request.llm_model,
        market_type=request.market_type.value,
        progress_callback=progress_callback,
        backend_url=request.backend_url,
        quick_think_model=request.quick_think_model,
        deep_think_model=request.deep_think_model,
    )

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, runner)

    response_data = {
        "success": bool(results.get("success")),
        "results": results,
        "progress": progress_events,
    }

    if results.get("success"):
        response_data["formatted_results"] = format_analysis_results(results)
    else:
        error_msg = results.get("error")
        if error_msg:
            response_data["errors"] = [error_msg]

    return AnalysisResponse(**response_data)
