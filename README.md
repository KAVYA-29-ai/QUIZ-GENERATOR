# Quiz Generator — Netlify prototype

Minimal prototype: React (Vite) frontend + one Netlify Function for LLM calls and PDF extraction.

## Deploy steps
1. Create a new GitHub repo and push this project.
2. Go to Netlify, click "New site from Git" and connect your GitHub repo.
3. In Netlify site settings, set environment variables in Site → Build & deploy → Environment:
   - `GEMINI_API_KEY` (optional)
   - `GEMINI_MODEL` (optional)
   - `HF_API_KEY` (optional)
   - `HF_MODEL` (optional)
4. Deploy. Build command is `npm run build`, published folder is `dist`.

## Notes
- Netlify Functions on free tier have short execution timeouts. The function includes a local fallback question generator to guarantee returned content.
- For PDF extraction we use `pdf-parse` on the serverless function.
