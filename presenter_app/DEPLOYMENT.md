# Deploying CMSC 173 Presenter App to Vercel

This guide walks you through deploying the Flask-based presenter app to Vercel.

## Prerequisites

1. A Vercel account (sign up at https://vercel.com)
2. Vercel CLI installed: `npm install -g vercel`
3. Git repository with your code
4. Supabase project (optional, for analytics and group portal features)

## Environment Variables

Before deploying, you'll need to set up the following environment variables in your Vercel project:

### Required for Supabase Features:
- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_ANON_KEY` - Your Supabase anonymous key

### Required for Admin Portal:
- `FLASK_SECRET_KEY` - A random secret key for session management
- `ADMIN_USERNAME` - Admin username for group portal management
- `ADMIN_PASSWORD` - Admin password for group portal management

## Deployment Steps

### Method 1: Deploy via Vercel CLI (Recommended)

1. **Install Vercel CLI** (if not already installed):
   ```bash
   npm install -g vercel
   ```

2. **Navigate to the presenter_app directory**:
   ```bash
   cd presenter_app
   ```

3. **Login to Vercel**:
   ```bash
   vercel login
   ```

4. **Deploy**:
   ```bash
   vercel
   ```

   Follow the prompts:
   - Set up and deploy? **Y**
   - Which scope? Select your account
   - Link to existing project? **N** (first time) or **Y** (subsequent deploys)
   - What's your project's name? **cmsc173-presenter** (or your preferred name)
   - In which directory is your code located? **./** (current directory)

5. **Set environment variables**:
   ```bash
   vercel env add SUPABASE_URL
   vercel env add SUPABASE_ANON_KEY
   vercel env add FLASK_SECRET_KEY
   vercel env add ADMIN_USERNAME
   vercel env add ADMIN_PASSWORD
   ```

   Or set them via the Vercel dashboard:
   - Go to your project settings
   - Navigate to "Environment Variables"
   - Add each variable for Production, Preview, and Development

6. **Deploy to production**:
   ```bash
   vercel --prod
   ```

### Method 2: Deploy via GitHub Integration

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push origin main
   ```

2. **Import project in Vercel**:
   - Go to https://vercel.com/new
   - Click "Import Git Repository"
   - Select your repository
   - Configure project:
     - **Root Directory**: `presenter_app`
     - **Framework Preset**: Other
     - **Build Command**: (leave empty)
     - **Output Directory**: (leave empty)

3. **Add environment variables**:
   - In the project settings, add all required environment variables
   - Make sure to add them for all environments (Production, Preview, Development)

4. **Deploy**:
   - Click "Deploy"
   - Vercel will automatically deploy on every push to your main branch

## Vercel Configuration

The app is configured via `vercel.json`:

```json
{
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python",
      "config": { "maxLambdaSize": "15mb", "runtime": "python3.9" }
    }
  ],
  "routes": [
    {
      "src": "/static/(.*)",
      "dest": "/static/$1"
    },
    {
      "src": "/images/(.*)",
      "dest": "/images/$1"
    },
    {
      "src": "/(.*)",
      "dest": "api/index.py"
    }
  ]
}
```

This configuration:
- Builds the Flask app as a serverless function
- Routes static files and images appropriately
- Routes all other requests to the Flask app

## Post-Deployment

### Verify Deployment

1. Visit your Vercel deployment URL (e.g., `https://cmsc173-presenter.vercel.app`)
2. Check that the homepage loads with all modules listed
3. Test a module page to ensure HTML presentations load
4. If using Supabase, verify analytics are tracking

### Troubleshooting

**Issue: 500 Internal Server Error**
- Check Vercel function logs in the dashboard
- Verify all environment variables are set correctly
- Ensure Supabase credentials are valid (if using Supabase features)

**Issue: Static files (images) not loading**
- Verify images are in the `images/` directory
- Check that .vercelignore doesn't exclude the images folder
- Ensure image paths in HTML templates use `/images/filename`

**Issue: Module views not incrementing**
- Verify Supabase environment variables are set
- Check that the `module_views` table exists in Supabase
- Ensure the `increment_module_view` function is created in Supabase

**Issue: Group Portal not working**
- Verify admin credentials environment variables are set
- Check Supabase tables: `groups`, `group_members`, `group_documents`
- Ensure Flask secret key is set for session management

## Supabase Database Setup

If you're using Supabase for analytics and group portal features, run these SQL commands in your Supabase SQL Editor:

### 1. Module Views Table
```sql
CREATE TABLE module_views (
  module_number INT PRIMARY KEY,
  view_count INT DEFAULT 0
);
```

### 2. Increment View Function
```sql
CREATE OR REPLACE FUNCTION increment_module_view(module_id INT)
RETURNS VOID AS $$
BEGIN
  INSERT INTO module_views (module_number, view_count)
  VALUES (module_id, 1)
  ON CONFLICT (module_number)
  DO UPDATE SET view_count = module_views.view_count + 1;
END;
$$ LANGUAGE plpgsql;
```

### 3. Group Portal Tables
```sql
-- Groups table
CREATE TABLE groups (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  group_name TEXT NOT NULL,
  project_title TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Group members table
CREATE TABLE group_members (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  group_id UUID REFERENCES groups(id) ON DELETE CASCADE,
  member_name TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Group documents table
CREATE TABLE group_documents (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  group_id UUID REFERENCES groups(id) ON DELETE CASCADE,
  document_title TEXT NOT NULL,
  file_path TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);
```

## Updating the Deployment

### Via CLI:
```bash
cd presenter_app
vercel --prod
```

### Via GitHub:
- Simply push changes to your main branch
- Vercel will automatically redeploy

## Custom Domain (Optional)

To use a custom domain:

1. Go to your project settings in Vercel
2. Navigate to "Domains"
3. Add your custom domain
4. Update your DNS records as instructed by Vercel

## Monitoring

- **Function Logs**: View real-time logs in the Vercel dashboard under "Deployments" > "Functions"
- **Analytics**: Vercel provides built-in analytics for page views and performance
- **Supabase Logs**: Check Supabase dashboard for database queries and errors

## Security Notes

1. **Never commit** `.env` file to Git
2. **Rotate secrets** regularly (Flask secret key, admin credentials)
3. **Use environment variables** for all sensitive data
4. **Enable Supabase Row Level Security (RLS)** for production databases
5. **Consider rate limiting** for API endpoints in production

## Support

For issues specific to:
- **Vercel deployment**: https://vercel.com/docs
- **Supabase setup**: https://supabase.com/docs
- **Flask configuration**: https://flask.palletsprojects.com/

## Cost

- **Vercel**: Free tier includes 100GB bandwidth and serverless function invocations
- **Supabase**: Free tier includes 500MB database, 1GB file storage, 2GB bandwidth
- Most educational use cases will fit within free tiers
