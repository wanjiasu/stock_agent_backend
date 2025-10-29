import asyncio
import os
from datetime import date, datetime
from enum import Enum
from functools import partial
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from fastapi.middleware.cors import CORSMiddleware

from cli.models import AnalystType
from web.utils.analysis_runner import (
    format_analysis_results,
    run_stock_analysis,
    validate_analysis_params,
)
from tradingagents.config.database_manager import (
    get_mongodb_client,
    get_redis_client,
    get_database_manager,
)

# 确保加载环境变量（兼容CLI/Web配置）
load_dotenv(override=True)

app = FastAPI(
    title="TradingAgents API",
    description="通过 FastAPI 暴露 TradingAgents 股票分析能力",
    version="0.1.0",
)

def _parse_list_env(var_name: str, default: List[str]) -> List[str]:
    value = os.getenv(var_name)
    if not value:
        return default
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items or default

# CORS 设置：默认允许本地前端；可通过环境变量覆盖
_default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]

allow_origins = _parse_list_env("CORS_ALLOW_ORIGINS", _default_origins)

methods_env = os.getenv("CORS_ALLOW_METHODS", "*").strip()
allow_methods = (
    ["*"] if methods_env == "*" else [m.strip() for m in methods_env.split(",") if m.strip()]
)

headers_env = os.getenv("CORS_ALLOW_HEADERS", "*").strip()
allow_headers = (
    ["*"] if headers_env == "*" else [h.strip() for h in headers_env.split(",") if h.strip()]
)

allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() in (
    "true",
    "1",
    "yes",
    "on",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=allow_methods,
    allow_headers=allow_headers,
)

# ===== Worker 联动（可选，通过环境变量启用） =====
# 设置 RUN_WORKER_WITH_SERVER=true 时，uvicorn 启动会同时拉起分析 worker。
# 注意：在使用 --reload 或多进程模式时可能造成重复启动，生产环境建议独立进程或 Docker。
WORKER_POPEN = None

@app.on_event("startup")
async def _start_worker_with_server():
    enabled = os.getenv("RUN_WORKER_WITH_SERVER", "false").lower() in ("true", "1", "yes", "on")
    if not enabled:
        return
    try:
        import subprocess, sys
        from pathlib import Path
        backend_root = Path(__file__).resolve().parents[1]
        worker_script = backend_root / "workers" / "analysis_worker.py"
        if not worker_script.exists():
            print(f"[Worker] 未找到 {worker_script}")
            return
        env = os.environ.copy()
        # 保持与当前环境一致（Mongo/Redis 开关/主机等）
        global WORKER_POPEN
        WORKER_POPEN = subprocess.Popen(
            [sys.executable, "-u", str(worker_script)],
            cwd=str(backend_root),
            env=env,
        )
        print(f"[Worker] 已启动 PID={WORKER_POPEN.pid}")
    except Exception as e:
        print(f"[Worker] 启动失败: {e}")

@app.on_event("shutdown")
async def _stop_worker_with_server():
    global WORKER_POPEN
    if WORKER_POPEN:
        try:
            WORKER_POPEN.terminate()
            WORKER_POPEN.wait(timeout=10)
            print("[Worker] 已停止")
        except Exception:
            try:
                WORKER_POPEN.kill()
                print("[Worker] 已强制停止")
            except Exception as e:
                print(f"[Worker] 停止失败: {e}")
        WORKER_POPEN = None


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

# 新增：带邮箱的排队请求模型
class AnalyzeAndEmailRequest(AnalysisRequest):
    notify_email: Optional[str] = Field(None, description="接收报告的邮箱")

class EnqueueResponse(BaseModel):
    task_id: str
    status: str
    message: str


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


# 新增：分析排队并邮件通知端点
@app.post("/analyze-and-email", response_model=EnqueueResponse)
async def analyze_and_email(request: AnalyzeAndEmailRequest):
    """
    接收分析请求与邮箱，生成唯一 task_id，创建MongoDB任务记录（queued），
    将 task_id 推入 Redis 队列，成功后返回 task_id。
    """
    import uuid

    # 预验证参数，保证基础合法性（与 /analyze 一致）
    analysts = [
        analyst.value if isinstance(analyst, AnalystType) else analyst
        for analyst in request.analysts
    ]
    analysis_date_str = request.analysis_date.strftime("%Y-%m-%d")

    is_valid, validation_errors = validate_analysis_params(
        stock_symbol=request.stock_symbol,
        analysis_date=analysis_date_str,
        analysts=analysts,
        research_depth=request.research_depth,
        market_type=request.market_type.value,
    )
    if not is_valid:
        raise HTTPException(status_code=422, detail=validation_errors)

    # 生成唯一 task_id
    task_id = uuid.uuid4().hex

    # MongoDB 连接
    mongodb_client = get_mongodb_client()
    if mongodb_client is None:
        raise HTTPException(status_code=503, detail="MongoDB不可用或未启用")

    db_name = get_database_manager().mongodb_config.get("database", "tradingagents")
    db = mongodb_client[db_name]
    tasks_coll = db["analysis_tasks"]

    now = datetime.utcnow()
    record = {
        "task_id": task_id,
        "status": "queued",
        "created_time": now,
        "updated_time": now,
        "request": {
            "ticker": request.stock_symbol,
            "analysis_date": analysis_date_str,
            "analysts": analysts,
            "research_depth": request.research_depth,
            "llm_provider": request.llm_provider,
            "llm_model": request.llm_model,
            "market_type": request.market_type.value,
            "notify_email": request.notify_email,
        },
    }

    try:
        insert_res = tasks_coll.insert_one(record)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建任务记录失败: {e}")

    if not getattr(insert_res, "inserted_id", None):
        raise HTTPException(status_code=500, detail="任务记录插入未确认")

    # Redis 入队
    redis_client = get_redis_client()
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis不可用或未启用")

    queue_name = os.getenv("ANALYSIS_TASK_QUEUE", "analysis_tasks_queue")
    try:
        queue_len = redis_client.rpush(queue_name, task_id)
    except Exception as e:
        # 入队失败，回滚任务状态为 failed
        try:
            tasks_coll.update_one({"task_id": task_id}, {"$set": {"status": "failed", "updated_time": datetime.utcnow()}})
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"任务入队失败: {e}")

    # 更新任务记录（标记已入队）
    try:
        tasks_coll.update_one(
            {"task_id": task_id},
            {"$set": {"enqueued": True, "queue": queue_name, "updated_time": datetime.utcnow()}}
        )
    except Exception:
        pass

    return EnqueueResponse(task_id=task_id, status="queued", message="任务已进入队列，请稍后")

# 旧的占位端点已移除，实际实现见上方 /analyze-and-email。

@app.get("/reports/by-task/{task_id}")
def get_report_by_task(task_id: str):
    """
    根据 task_id 从 MongoDB 的 analysis_reports 集合读取报告数据。
    返回包含元数据与 reports（Markdown 文档集合）。
    """
    mongodb_client = get_mongodb_client()
    if mongodb_client is None:
        raise HTTPException(status_code=503, detail="MongoDB不可用或未启用")

    db_name = get_database_manager().mongodb_config.get("database", "tradingagents")
    db = mongodb_client[db_name]
    reports_coll = db["analysis_reports"]

    try:
        doc = reports_coll.find_one({"task_id": task_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询MongoDB失败: {e}")

    if not doc:
        raise HTTPException(status_code=404, detail="未找到该task_id对应的报告")

    # 统一时间戳
    def to_ts(v):
        try:
            return v.timestamp() if hasattr(v, "timestamp") else v
        except Exception:
            return None

    reports = doc.get("reports") or {}

    return {
        "task_id": doc.get("task_id"),
        "analysis_id": doc.get("analysis_id"),
        "stock_symbol": doc.get("stock_symbol"),
        "analysis_date": doc.get("analysis_date"),
        "analysts": doc.get("analysts", []),
        "research_depth": doc.get("research_depth", 1),
        "status": doc.get("status", "completed"),
        "timestamp": to_ts(doc.get("timestamp")),
        "reports": reports,
    }
