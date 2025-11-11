# Image Detection Backend Service

Standalone Python backend service for image tampering detection. Deploy this separately on Railway.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   export IMAGE_DETECTION_API_USER=your_api_user
   export IMAGE_DETECTION_API_SECRET=your_api_secret
   export PORT=5000
   ```

3. **Run locally:**
   ```bash
   python app.py
   ```

## Deployment on Railway

### Option 1: Using Nixpacks (Recommended)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-backend-repo-url>
   git push -u origin main
   ```

2. **Deploy on Railway:**
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your backend repository
   - Railway will auto-detect Python and deploy

3. **If you get Python version errors:**
   - Go to your Railway project → Settings
   - Under "Build", change builder to **"Nixpacks"** (instead of Railpack)
   - Or remove `runtime.txt` and let Railway auto-detect Python version

4. **Set Environment Variables in Railway:**
   - Go to your project → Variables
   - Add:
     - `IMAGE_DETECTION_API_USER`
     - `IMAGE_DETECTION_API_SECRET`
     - `PORT` (Railway sets this automatically)

5. **Get your service URL:**
   - Railway will provide a URL like: `https://your-service.railway.app`
   - Copy this URL for your frontend configuration

### Option 2: Using Dockerfile

If Nixpacks doesn't work, Railway can use the Dockerfile:

1. **In Railway project settings:**
   - Go to Settings → Build
   - Change builder to **"Dockerfile"**
   - Railway will use the provided Dockerfile

2. **Deploy as usual**

## API Endpoints

- `GET /health` - Health check
- `POST /detect` - Image detection (multipart/form-data with 'image' field)

## Troubleshooting

### Error: '$PORT' is not a valid port number

If you see this error, it means Railway isn't expanding the PORT variable. The `start.sh` script should fix this. Make sure:

1. `start.sh` is executable (it should be after git push)
2. Procfile uses: `web: bash start.sh`
3. If still having issues, Railway should auto-detect and use the start script

### Alternative: Use Python start script

If bash script doesn't work, you can use the Python alternative:

1. Change Procfile to:
   ```
   web: python start.py
   ```

2. The `start.py` script reads PORT from environment and starts gunicorn

## Testing

```bash
# Health check
curl https://your-service.railway.app/health

# Image detection
curl -X POST https://your-service.railway.app/detect \
  -F "image=@test-image.jpg"
```

