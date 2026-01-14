module.exports = {
  apps: [
    {
      name: 'tradingagents-api',
      script: '.venv/bin/python',
      args: '-m uvicorn api.server:app --host 0.0.0.0 --port 8000'
    }
  ]
};
