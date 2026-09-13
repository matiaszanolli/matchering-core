import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import { execSync } from 'node:child_process'
import { vendorChunk } from './vite.manualChunks'

// Function to get current commit hash (called fresh each time in dev mode)
function getCommitId(): string {
  try {
    return execSync('git rev-parse --short HEAD', { encoding: 'utf8' }).trim()
  } catch {
    return 'unknown'
  }
}

export default defineConfig(({ mode }) => {
  // Get commit at config time for build mode
  const commitIdAtBuildTime = getCommitId()

  return {
    plugins: [
      react(),
      {
        name: 'inject-commit-id',
        transformIndexHtml: {
          order: 'pre',
          handler(html: string) {
            // In dev mode, get fresh commit hash for each request
            // In build mode, use the commit from config time
            const commitId = mode === 'serve' ? getCommitId() : commitIdAtBuildTime
            return html.replace('%VITE_COMMIT_ID%', commitId)
          },
        },
      },
      {
        name: 'fix-vendor-loading-order',
        transformIndexHtml: {
          order: 'post',
          handler(html: string) {
            if (mode === 'serve') return html

            // Find vendor and app script tags
            const vendorMatch = html.match(/<link rel="modulepreload"[^>]*href="\/[^"]*vendor[^"]*"[^>]*>/)
            const appScriptMatch = html.match(/<script type="module"[^>]*src="\/[^"]*\/(assets\/)?index[^"]*"[^>]*><\/script>/)

            if (!vendorMatch || !appScriptMatch) return html

            // Extract the vendor filename
            const vendorHref = vendorMatch[0].match(/href="([^"]*)"/)?.[1]
            if (!vendorHref) return html

            // Extract app src
            const appSrc = appScriptMatch[0].match(/src="([^"]*)"/)?.[1] || '/assets/index.js'

            // Strategy: Remove modulepreload link completely to prevent race condition.
            // Instead, use a loader script that explicitly loads vendor first,
            // then loads the app which has static imports from vendor.
            //
            // The key: by the time `import(appSrc)` executes, vendor is already
            // fully loaded and initialized, so the static imports in app.js
            // will reuse the vendor module instead of triggering a new load.
            const loaderScript = `<script type="module">
  (async () => {
    try {
      // Load vendor module first to guarantee it's in the module cache
      const vendor = await import('${vendorHref}');

      // Wait multiple event loop cycles to ensure all vendor initialization is complete
      // Vendor module contains React, MUI, emotion which register themselves on load
      for (let i = 0; i < 5; i++) {
        await new Promise(r => setTimeout(r, 0));
      }

      // Now load app - its static imports will reuse vendor from module cache
      const appModule = await import('${appSrc}');

    } catch (err) {
      console.error('[loader] Fatal loading error:', err);
      const msg = (err && err.message) || 'Unknown error';
      const stack = (err && err.stack) || '';
      console.error('[loader] Full error:', {msg, stack});
      document.documentElement.innerHTML = '<body style="background:#1a1a1a;color:#fff;font-family:sans-serif;margin:0;padding:0;display:flex;align-items:center;justify-content:center;height:100vh"><div style="text-align:center;max-width:700px;padding:40px"><h1 style="color:#ff6b6b;margin:0 0 20px;font-size:28px">Application Error</h1><p style="color:#ccc;margin:0 0 20px;font-size:16px">Failed to load modules</p><p style="color:#999;margin:0 0 20px;font-size:14px;font-family:monospace">' + msg + '</p><details style="text-align:left;margin:30px 0;padding:20px;background:#222;border:1px solid #444;border-radius:6px"><summary style="color:#667eea;cursor:pointer;font-weight:bold;margin-bottom:10px">Stack Trace</summary><pre style="color:#888;font-size:11px;overflow-x:auto;margin:0;white-space:pre-wrap;word-break:break-word;line-height:1.4">' + stack.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</pre></details><hr style="border:none;border-top:1px solid #333;margin:20px 0"><p style="color:#666;font-size:12px;margin:20px 0">Vendor: ${vendorHref}<br>App: ${appSrc}</p></div></body>';
    }
  })();
</script>`

            return html
              .replace(/<link rel="modulepreload"[^>]*href="\/[^"]*vendor[^"]*"[^>]*>/, '') // Remove modulepreload to eliminate race
              .replace(/<script type="module"[^>]*src="\/[^"]*\/(assets\/)?index[^"]*"[^>]*><\/script>/, loaderScript)
          },
        },
      },
    ],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    define: {
      'process.env.VITE_COMMIT_ID': `"${commitIdAtBuildTime}"`,
    },
    server: {
      port: 3000,
      open: false,
      proxy: {
        '/api': {
          target: 'http://localhost:8765',
          changeOrigin: true,
        },
        '/ws': {
          target: 'ws://localhost:8765',
          ws: true,
          changeOrigin: true,
        },
      },
    },
    esbuild: {
      // Strip console.* and debugger statements from production bundles.
      // The injected loader script (raw HTML) is unaffected — its two
      // console.error calls in the fatal-error catch block are intentional.
      drop: mode === 'production' ? ['console', 'debugger'] : [],
    },
    build: {
      target: 'esnext',
      outDir: 'dist',
      sourcemap: false,
      rollupOptions: {
        output: {
          // Separate vendor chunk for better module initialization order.
          // Defined in vite.manualChunks.ts so it is unit-testable — read the
          // rationale there before changing it (#4697).
          manualChunks: vendorChunk,
          chunkFileNames: '[name]-[hash].js',
        },
      },
      // The `vendor` chunk is intentionally large (~705 kB raw / ~215 kB gzip
      // as of #4697) because the rule above deliberately front-loads React,
      // MUI and Emotion. Warning about it on every build trained everyone to
      // ignore the warning; this limit is set just above the current vendor
      // size so real growth still surfaces. Raise it only with a measurement.
      chunkSizeWarningLimit: 750,
    },
  }
})
