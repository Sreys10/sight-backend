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

3. **Set Environment Variables in Railway:**
   - Go to your project → Variables
   - Add:
     - `IMAGE_DETECTION_API_USER`
     - `IMAGE_DETECTION_API_SECRET`
     - `PORT` (Railway sets this automatically)

4. **Get your service URL:**
   - Railway will provide a URL like: `https://your-service.railway.app`
   - Copy this URL for your frontend configuration

## API Endpoints

- `GET /health` - Health check
- `POST /detect` - Image detection (multipart/form-data with 'image' field)

## Testing

```bash
# Health check
curl https://your-service.railway.app/health

# Image detection
curl -X POST https://your-service.railway.app/detect \
  -F "image=@test-image.jpg"
```

