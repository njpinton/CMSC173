# Vercel Deployment Setup Guide

## Environment Variables Required

After logging into Vercel, you need to set these environment variables in your project settings:

### 1. Security Variables (Required)
```
FLASK_SECRET_KEY=your_secure_secret_key_here
ADMIN_USERNAME=admin
ADMIN_PASSWORD_HASH=hashed_password
```

### 2. Supabase Variables (Required)
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here
```

### 3. CORS Variables (Optional - defaults to localhost)
```
ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

## Steps to Configure in Vercel

### Option A: Using Vercel CLI
```bash
# Login to Vercel
vercel login

# Set environment variables
vercel env add FLASK_SECRET_KEY
vercel env add ADMIN_USERNAME
vercel env add ADMIN_PASSWORD_HASH
vercel env add SUPABASE_URL
vercel env add SUPABASE_ANON_KEY
vercel env add ALLOWED_ORIGINS

# Deploy
vercel deploy --prod
```

### Option B: Using Vercel Dashboard

1. Go to [https://vercel.com/dashboard](https://vercel.com/dashboard)
2. Select your `CMSC173` project
3. Go to **Settings** → **Environment Variables**
4. Add each variable:
   - Name: `FLASK_SECRET_KEY`
   - Value: (generate a secure key)
   - Environments: Check all (Production, Preview, Development)
   - Click "Save"

5. Repeat for all variables above

## Generating Required Values

### Generate FLASK_SECRET_KEY
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Generate ADMIN_PASSWORD_HASH
```bash
python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('your_password'))"
```

## Getting Supabase Credentials

1. Go to [supabase.com/dashboard](https://supabase.com/dashboard)
2. Select your project
3. Go to **Settings** → **API**
4. Copy:
   - `URL` → SUPABASE_URL
   - `anon key` → SUPABASE_ANON_KEY

## Verifying Deployment

After setting environment variables:

1. Trigger a new deployment:
   ```bash
   vercel deploy --prod
   ```

2. Check the deployment:
   - Go to your Vercel project dashboard
   - Click on the latest deployment
   - Check the build logs for any errors

3. Test the app:
   ```bash
   curl https://your-vercel-domain.vercel.app/
   ```

4. Expected result: You should see the CMSC 173 module listing page (HTML)

## Troubleshooting

### Issue: "FLASK_SECRET_KEY environment variable is not set"
- **Cause:** Environment variable not set in Vercel
- **Solution:** Add FLASK_SECRET_KEY to Vercel Environment Variables and redeploy

### Issue: "Supabase not configured"
- **Cause:** SUPABASE_URL or SUPABASE_ANON_KEY missing
- **Solution:** Verify both variables are set in Vercel dashboard

### Issue: Deployment succeeds but app returns 500
1. Check Vercel Function logs:
   - Dashboard → Deployments → Click latest → Logs
2. Look for error messages
3. Verify all environment variables are set

## Environment Variables Summary

| Variable | Required | Example |
|----------|----------|---------|
| FLASK_SECRET_KEY | ⚠️ Recommended | `SFMyNTQ2NzU4OTAxMjM0NTY...` |
| ADMIN_USERNAME | Yes* | `admin` |
| ADMIN_PASSWORD_HASH | Yes* | `scrypt:32768:8:1$...` |
| SUPABASE_URL | Yes | `https://abc.supabase.co` |
| SUPABASE_ANON_KEY | Yes | `eyJhbGciOiJIUzI1NiIs...` |
| ALLOWED_ORIGINS | No | `https://yourdomain.com` |

*Only if using admin login feature

## Post-Deployment Verification

### Test Endpoints

**Home Page:**
```bash
curl https://your-domain.vercel.app/
```
Expected: HTML with module listing

**Module View:**
```bash
curl https://your-domain.vercel.app/module/0
```
Expected: Module 0 HTML (if template exists)

**Favicon (no-op):**
```bash
curl https://your-domain.vercel.app/favicon.ico
```
Expected: 204 No Content

**API Groups (if configured):**
```bash
curl https://your-domain.vercel.app/api/groups
```
Expected: JSON response (200 if Supabase configured, 500 if not)

## Security Checklist

- [ ] FLASK_SECRET_KEY set and unique per environment
- [ ] ADMIN_PASSWORD_HASH is hashed (not plain text)
- [ ] SUPABASE_ANON_KEY is the anonymous key (not service role)
- [ ] ALLOWED_ORIGINS set to your production domain
- [ ] No secrets committed to git
- [ ] Environment variables not logged in build output

## Contact & Support

For issues:
1. Check Vercel Function logs
2. Review SECURITY_FIXES.md for implementation details
3. Verify all environment variables are set
4. Check Supabase status page for database issues

**Last Updated:** November 16, 2025
