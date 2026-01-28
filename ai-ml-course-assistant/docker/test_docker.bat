@echo off
echo.
echo ======================================
echo Docker Test - Safe Build
echo ======================================
echo.
echo Your production DB is SAFE!
echo Test uses: docker_test_data/
echo Production data: data/ (not mounted)
echo.
echo Building Docker image...
echo.

docker compose -f docker-compose.test.yml build --no-cache

if %errorlevel% neq 0 (
    echo.
    echo ❌ Build failed!
    pause
    exit /b 1
)

echo.
echo ✅ Build successful!
echo.
echo Starting test container...
echo.

docker compose -f docker-compose.test.yml up -d

if %errorlevel% neq 0 (
    echo.
    echo ❌ Failed to start container!
    pause
    exit /b 1
)

echo.
echo ✅ Container started!
echo.
echo Waiting for health check (30 seconds)...
timeout /t 30 /nobreak

echo.
echo Opening browser: http://localhost:8502
start http://localhost:8502

echo.
echo ======================================
echo To view logs:
echo   docker compose -f docker-compose.test.yml logs -f streamlit-test
echo.
echo To stop:
echo   docker compose -f docker-compose.test.yml down
echo ======================================
echo.

pause
