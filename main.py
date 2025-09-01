"""
애플리케이션 엔트리 포인트

새로운 구조의 FastAPI 애플리케이션을 실행합니다.
"""

from app.main import app

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

