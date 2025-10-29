import sys
from datetime import datetime
from dotenv import load_dotenv
from tradingagents.config.database_manager import get_mongodb_client, get_database_manager

# 载入 .env 配置，保持与API/Worker一致
load_dotenv(override=True)

def main(task_id: str):
    client = get_mongodb_client()
    if client is None:
        print("MongoClient=None")
        return
    db_name = get_database_manager().mongodb_config.get("database", "tradingagents")
    db = client[db_name]
    coll = db["analysis_tasks"]
    doc = coll.find_one({"task_id": task_id})
    if doc is None:
        print("Doc=None")
        return
    def fmt_ts(v):
        try:
            if hasattr(v, "timestamp"):
                return datetime.fromtimestamp(v.timestamp()).isoformat()
        except Exception:
            pass
        return str(v)
    progress = doc.get("progress") or []
    # 仅显示最近的3条进度
    recent = progress[-3:] if len(progress) > 3 else progress
    out = {
        "task_id": doc.get("task_id"),
        "status": doc.get("status"),
        "queue": doc.get("queue"),
        "enqueued": doc.get("enqueued"),
        "updated_time": fmt_ts(doc.get("updated_time")),
        "error": doc.get("error"),
        "recent_progress": [{"message": p.get("message"), "step": p.get("step"), "total": p.get("total_steps")} for p in recent],
    }
    print(out)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/tmp_query_task.py <task_id>")
        sys.exit(1)
    main(sys.argv[1])