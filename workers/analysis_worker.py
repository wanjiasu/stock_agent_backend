#!/usr/bin/env python3
"""
Analysis Worker
- Blockingly consume task_id from Redis
- Load task details from MongoDB
- Update task status to processing
- Run stock analysis and save report to MongoDB (include task_id)
- Send email via SMTP with report URL
- Mark task completed or failed with error details
"""

import os
import sys
import time
import smtplib
import traceback
from ssl import create_default_context
from email.message import EmailMessage
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from pathlib import Path

# Ensure project root in path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("analysis_worker")

from tradingagents.config.database_manager import (
    get_mongodb_client,
    get_redis_client,
    get_database_manager,
)

from web.utils.analysis_runner import run_stock_analysis, validate_analysis_params
from web.utils.mongodb_report_manager import mongodb_report_manager


# ---- SMTP helpers ----

def _env_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def send_report_email(to_email: str, report_url: str) -> None:
    """Send the report URL to user via SMTP. Raises on failure."""
    if not to_email:
        raise ValueError("notify_email is empty")

    mail_server = os.getenv("MAIL_SERVER")
    mail_port = int(os.getenv("MAIL_PORT", "465"))
    mail_username = os.getenv("MAIL_USERNAME")
    mail_password = os.getenv("MAIL_PASSWORD")
    mail_from = os.getenv("MAIL_FROM", mail_username)
    use_tls = _env_bool(os.getenv("MAIL_TLS"), default=False)
    use_ssl = _env_bool(os.getenv("MAIL_SSL"), default=True)

    if not mail_server or not mail_username or not mail_password:
        raise RuntimeError("SMTP config incomplete: MAIL_SERVER/MAIL_USERNAME/MAIL_PASSWORD required")

    msg = EmailMessage()
    msg["Subject"] = "TradingAgents 报告已生成"
    msg["From"] = mail_from
    msg["To"] = to_email
    msg.set_content(
        f"您好，您的股票分析报告已生成。\n\n"
        f"报告地址：{report_url}\n\n"
        f"如无法点击，请复制链接到浏览器打开。\n"
    )

    if use_ssl:
        context = create_default_context()
        with smtplib.SMTP_SSL(mail_server, mail_port, context=context) as server:
            server.login(mail_username, mail_password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(mail_server, mail_port) as server:
            server.ehlo()
            if use_tls:
                server.starttls(context=create_default_context())
                server.ehlo()
            server.login(mail_username, mail_password)
            server.send_message(msg)


# ---- Mongo helpers ----

def get_db_and_collections():
    mongo_client = get_mongodb_client()
    if mongo_client is None:
        raise RuntimeError("MongoDB client unavailable")
    db_name = get_database_manager().mongodb_config.get("database", "tradingagents")
    db = mongo_client[db_name]
    tasks_coll = db["analysis_tasks"]
    reports_coll = db["analysis_reports"]
    return db, tasks_coll, reports_coll


def update_task_status(tasks_coll, task_id: str, status: str, extra_set: Optional[Dict[str, Any]] = None) -> None:
    doc: Dict[str, Any] = {"status": status, "updated_time": datetime.utcnow()}
    if extra_set:
        doc.update(extra_set)
    tasks_coll.update_one({"task_id": task_id}, {"$set": doc})


def push_progress(tasks_coll, task_id: str, message: str, step: Optional[int] = None, total_steps: Optional[int] = None) -> None:
    try:
        tasks_coll.update_one(
            {"task_id": task_id},
            {"$push": {"progress": {"message": message, "step": step, "total_steps": total_steps, "ts": datetime.utcnow()}}}
        )
    except Exception:
        # Non-critical
        pass


# ---- Task processor ----

def process_task(task_id: str) -> None:
    logger.info(f"📥 处理任务: {task_id}")
    db, tasks_coll, _reports_coll = get_db_and_collections()

    task_doc = tasks_coll.find_one({"task_id": task_id})
    if not task_doc:
        raise RuntimeError(f"任务不存在: {task_id}")

    # Update status to processing
    update_task_status(tasks_coll, task_id, "processing", {"started_time": datetime.utcnow()})

    request = task_doc.get("request", {})
    ticker = request.get("ticker")
    analysis_date = request.get("analysis_date")  # string like 'YYYY-MM-DD'
    analysts = request.get("analysts", [])
    research_depth = request.get("research_depth", 1)
    llm_provider = request.get("llm_provider")
    llm_model = request.get("llm_model")
    market_type = request.get("market_type", "美股")
    backend_url = request.get("backend_url")
    quick_think_model = request.get("quick_think_model")
    deep_think_model = request.get("deep_think_model")
    notify_email = request.get("notify_email")

    # Validate
    is_valid, validation_errors = validate_analysis_params(
        stock_symbol=ticker,
        analysis_date=analysis_date,
        analysts=analysts,
        research_depth=research_depth,
        market_type=market_type,
    )
    if not is_valid:
        update_task_status(tasks_coll, task_id, "failed", {"error": validation_errors})
        raise RuntimeError(f"参数校验失败: {validation_errors}")

    # Run analysis
    push_progress(tasks_coll, task_id, "开始执行股票分析...")
    try:
        results = run_stock_analysis(
            stock_symbol=ticker,
            analysis_date=analysis_date,
            analysts=analysts,
            research_depth=research_depth,
            llm_provider=llm_provider,
            llm_model=llm_model,
            market_type=market_type,
            progress_callback=lambda msg, step=None, total_steps=None: push_progress(tasks_coll, task_id, msg, step, total_steps),
            backend_url=backend_url,
            quick_think_model=quick_think_model,
            deep_think_model=deep_think_model,
        )
    except Exception as e:
        update_task_status(tasks_coll, task_id, "failed", {"error": str(e)})
        raise

    if not results or not results.get("success", False):
        err = results.get("error") if isinstance(results, dict) else "未知错误"
        update_task_status(tasks_coll, task_id, "failed", {"error": err})
        raise RuntimeError(f"分析失败: {err}")

    analysis_id = results.get("analysis_id")

    # Ensure task_id stored with analysis report (second upsert to add task_id)
    try:
        enriched = dict(results)
        enriched["task_id"] = task_id
        mongodb_report_manager.save_analysis_report(
            stock_symbol=ticker,
            analysis_results=enriched,
            reports={},  # no overwrite existing content
            analysis_id=analysis_id,
        )
    except Exception as e:
        # Saving task_id is important but not fatal to analysis itself; mark failed as per requirements if any step fails
        update_task_status(tasks_coll, task_id, "failed", {"error": f"保存报告失败: {str(e)}"})
        raise

    # Send email
    report_url = f"http://192.168.0.186:3000" + f"/report/{task_id}"
    push_progress(tasks_coll, task_id, "发送邮件通知用户...")
    try:
        send_report_email(notify_email, report_url)
    except Exception as e:
        update_task_status(tasks_coll, task_id, "failed", {"error": f"邮件发送失败: {str(e)}"})
        raise

    # Mark completed
    update_task_status(
        tasks_coll,
        task_id,
        "completed",
        {"completed_time": datetime.utcnow(), "report_url": report_url}
    )
    push_progress(tasks_coll, task_id, "任务已完成，报告和邮件发送成功。")
    logger.info(f"✅ 任务完成: {task_id}")


# ---- Main loop ----

def worker_loop():
    redis_client = get_redis_client()
    if redis_client is None:
        raise RuntimeError("Redis client unavailable")

    queue_name = os.getenv("ANALYSIS_TASK_QUEUE", "analysis_tasks_queue")
    logger.info(f"🚀 Worker启动，监听Redis队列: {queue_name}")

    while True:
        try:
            item = redis_client.blpop(queue_name, timeout=0)  # blocking
            if not item:
                continue
            _, task_id_bytes = item
            task_id = task_id_bytes.decode("utf-8") if isinstance(task_id_bytes, (bytes, bytearray)) else str(task_id_bytes)
            process_task(task_id)
        except Exception as e:
            logger.error(f"❌ 处理任务时发生错误: {e}")
            logger.error(traceback.format_exc())
            # 短暂休眠，避免异常导致空转狂刷
            time.sleep(1)


if __name__ == "__main__":
    try:
        worker_loop()
    except KeyboardInterrupt:
        logger.info("🛑 Worker已停止 (Ctrl+C)")