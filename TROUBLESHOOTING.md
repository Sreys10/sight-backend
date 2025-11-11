# Troubleshooting Guide

## "Not Found" Error

If you're seeing a "Not Found" error, here's how to diagnose and fix it:

### 1. Check if Backend is Running

Test the health endpoint:
```bash
curl https://your-service.railway.app/health
```

Expected response:
```json
{"status":"ok","service":"image-detection-backend","version":"1.0.0"}
```

### 2. Check Root Endpoint

Test the root endpoint:
```bash
curl https://your-service.railway.app/
```

Expected response:
```json
{
  "status": "ok",
  "service": "image-detection-backend",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "detect": "/detect"
  }
}
```

### 3. Common Issues

#### Issue: Backend URL is Wrong

**Check:**
- Verify your Railway service URL is correct
- Make sure there's no trailing slash
- Check environment variable `BACKEND_SERVICE_URL` in Vercel

**Fix:**
```bash
# In Vercel, set:
BACKEND_SERVICE_URL=https://your-service.railway.app
# NOT: https://your-service.railway.app/
```

#### Issue: Backend Not Deployed

**Check:**
- Go to Railway dashboard
- Check if service is running (green status)
- Check deployment logs for errors

**Fix:**
- If deployment failed, check logs
- Make sure all environment variables are set
- Redeploy if needed

#### Issue: CORS Error

**Check:**
- Open browser DevTools → Console
- Look for CORS errors

**Fix:**
- Backend already has CORS enabled
- Make sure frontend URL is allowed
- Check if backend is actually running

#### Issue: Wrong Route

**Check:**
- Frontend calls: `/api/detect-tampering`
- Backend expects: `/detect`
- Make sure frontend API route forwards to backend `/detect`

**Fix:**
- Verify `app/api/detect-tampering/route.ts` calls `${BACKEND_SERVICE_URL}/detect`

### 4. Testing Endpoints

#### Test Health Endpoint
```bash
curl https://your-service.railway.app/health
```

#### Test Root Endpoint
```bash
curl https://your-service.railway.app/
```

#### Test Detect Endpoint
```bash
curl -X POST https://your-service.railway.app/detect \
  -F "image=@test-image.jpg"
```

### 5. Check Railway Logs

1. Go to Railway dashboard
2. Click on your service
3. Go to "Deployments" tab
4. Click on latest deployment
5. Check logs for errors

### 6. Check Vercel Logs

1. Go to Vercel dashboard
2. Click on your project
3. Go to "Deployments"
4. Click on latest deployment
5. Check "Functions" tab for API route logs

### 7. Verify Environment Variables

**Backend (Railway):**
- `IMAGE_DETECTION_API_USER`
- `IMAGE_DETECTION_API_SECRET`
- `PORT` (auto-set by Railway)

**Frontend (Vercel):**
- `BACKEND_SERVICE_URL` (most important!)
- `MONGODB_URI`
- `MONGODB_DB_NAME`
- `NEXT_PUBLIC_APP_URL`

### 8. Common URL Mistakes

❌ Wrong:
```
BACKEND_SERVICE_URL=https://your-service.railway.app/
BACKEND_SERVICE_URL=http://your-service.railway.app
BACKEND_SERVICE_URL=your-service.railway.app
```

✅ Correct:
```
BACKEND_SERVICE_URL=https://your-service.railway.app
```

### 9. Debug Steps

1. **Test backend directly:**
   ```bash
   curl https://your-service.railway.app/health
   ```

2. **Check frontend API route:**
   - Open browser DevTools → Network tab
   - Upload an image
   - Check what URL is being called
   - Check response status

3. **Check backend logs:**
   - Railway dashboard → Service → Logs
   - Look for incoming requests
   - Check for errors

4. **Verify connection:**
   - Make sure backend URL is accessible
   - Check if Railway service is running
   - Verify no firewall blocking

### 10. Still Not Working?

1. **Restart Railway service:**
   - Railway dashboard → Service → Settings → Restart

2. **Redeploy:**
   - Push a new commit to trigger redeploy
   - Or manually redeploy in Railway

3. **Check service status:**
   - Railway dashboard should show green "Active" status
   - Check if service is paused or stopped

4. **Contact support:**
   - Railway: Check their status page
   - Vercel: Check their status page

