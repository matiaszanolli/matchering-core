const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron');
const { autoUpdater } = require('electron-updater');
const log = require('electron-log/main');
const { spawn, exec } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');
const { isSafeExternalUrl, isAllowedAppNavigation } = require('./url-safety');
const { buildBackendEnv } = require('./backend-env');

// Configure logging for auto-updater
log.transports.file.level = 'info';

// Owner-only log file (#4932). electron-log's file transport defaults to
// requesting mode 0o666, so the file lands at 0o666 & ~umask — 0o644 under the
// usual 0o022, i.e. world-readable. That is harmless under this app's
// single-user desktop threat model and while the file holds only update-check
// timestamps, but #4920 has since begun routing backend output through this
// same file, so its contents are no longer trivially uninteresting. 0o600
// costs nothing and stops the question recurring.
//
// Merged onto the transport's existing writeOptions rather than replaced:
// assigning a bare { mode } would drop the upstream flag:'a' and
// encoding:'utf8' defaults, truncating the log on every launch.
log.transports.file.writeOptions = {
  ...log.transports.file.writeOptions,
  mode: 0o600,
};

autoUpdater.logger = log;

/**
 * Single validated path to the OS shell for any navigation the renderer
 * tries to open externally — both setWindowOpenHandler call sites below
 * route through this instead of calling shell.openExternal directly (#4844).
 */
function openExternalSafely(url) {
  if (!isSafeExternalUrl(url)) {
    console.warn(`Blocked shell.openExternal for disallowed URL: ${url}`);
    return;
  }
  shell.openExternal(url);
}

class AuralisApp {
  constructor() {
    this.pythonProcess = null;
    this.mainWindow = null;
    this.backendReady = false;
    this.isQuitting = false;
    this.isDevelopment = !app.isPackaged;
    this.backendPort = 8765;
  }

  /**
   * Check if port is in use and kill any process using it
   * Cross-platform: works on Linux, macOS, and Windows
   */
  async cleanupPort() {
    return new Promise((resolve) => {
      console.log(`Checking if port ${this.backendPort} is available...`);

      const isWindows = process.platform === 'win32';
      const findCommand = isWindows
        ? `netstat -ano | findstr :${this.backendPort}`
        : `lsof -ti:${this.backendPort}`;

      exec(findCommand, (error, stdout, stderr) => {
        if (error || !stdout.trim()) {
          console.log(`✓ Port ${this.backendPort} is available`);
          resolve();
          return;
        }

        let pids = [];
        if (isWindows) {
          const lines = stdout.trim().split('\n');
          pids = lines
            .map(line => {
              const match = line.trim().match(/\s+(\d+)\s*$/);
              return match ? match[1] : null;
            })
            .filter(pid => pid);
          pids = [...new Set(pids)];
        } else {
          pids = stdout.trim().split('\n').filter(pid => pid);
        }

        if (pids.length === 0) {
          console.log(`✓ Port ${this.backendPort} is available`);
          resolve();
          return;
        }

        console.log(`⚠️ Found ${pids.length} process(es) using port ${this.backendPort}: ${pids.join(', ')}`);

        // Force-kill all processes (SIGKILL) for reliable cleanup
        const killPromises = pids.map(pid => {
          return new Promise((killResolve) => {
            const killCommand = isWindows ? `taskkill /F /PID ${pid}` : `kill -9 ${pid}`;
            exec(killCommand, (killError) => {
              if (killError) {
                console.error(`Failed to kill process ${pid}:`, killError.message);
              } else {
                console.log(`✓ Killed process ${pid}`);
              }
              killResolve();
            });
          });
        });

        Promise.all(killPromises).then(() => {
          // Poll until port is actually free (up to 5 seconds)
          const startTime = Date.now();
          const pollInterval = 250;
          const maxWait = 5000;

          const checkPort = () => {
            exec(findCommand, (err, out) => {
              if (err || !out.trim()) {
                console.log(`✓ Port ${this.backendPort} is free`);
                resolve();
              } else if (Date.now() - startTime > maxWait) {
                console.warn(`⚠️ Port ${this.backendPort} still in use after ${maxWait}ms, proceeding anyway`);
                resolve();
              } else {
                setTimeout(checkPort, pollInterval);
              }
            });
          };

          // Give initial kill a moment then start polling
          setTimeout(checkPort, 500);
        });
      });
    });
  }

  async startPythonBackend() {
    console.log('Starting Python backend...');
    console.log('Development mode:', this.isDevelopment);

    // First, ensure port is free
    await this.cleanupPort();

    let pythonCmd, pythonArgs, cwd;

    if (this.isDevelopment) {
      // Development: Run Python script directly
      pythonCmd = 'python';
      pythonArgs = [path.join(__dirname, '..', 'auralis-web', 'backend', 'main.py')];
      cwd = path.join(__dirname, '..');
    } else {
      // Production: Run Python backend from resources
      // Platform-specific backend selection
      const isWindows = process.platform === 'win32';
      const backendPath = path.join(process.resourcesPath, 'backend', 'auralis-backend');
      const backendExe = isWindows ? `${backendPath}.exe` : backendPath;
      const pythonScript = path.join(process.resourcesPath, 'backend', 'main.py');

      if (fs.existsSync(backendExe)) {
        // Use compiled binary if available
        pythonCmd = backendExe;
        pythonArgs = [];
        cwd = path.join(process.resourcesPath, 'backend');
      } else if (!isWindows && fs.existsSync(pythonScript)) {
        // macOS/Linux: Fallback to Python script from resources
        console.log('Binary not found, using Python script from resources');
        pythonCmd = 'python3';
        pythonArgs = [pythonScript];
        cwd = path.join(process.resourcesPath, 'backend');
      } else {
        // Error: Backend not available
        const errorMessage = isWindows
          ? `Auralis backend not found.\n\nPlease reinstall Auralis.\n\nAlternatively, install Python 3.13+:\nhttps://python.org`
          : `Backend not found. Tried:\n  Binary: ${backendExe}\n  Script: ${pythonScript}`;

        console.error(errorMessage);

        if (isWindows) {
          // Show user-friendly dialog for Windows
          dialog.showErrorBox('Backend Not Found', errorMessage);
        }

        throw new Error('Backend executable or script not found');
      }
    }

    console.log(`Executing: ${pythonCmd} ${pythonArgs.join(' ')}`);
    console.log(`Working directory: ${cwd}`);

    // Set up Python path to include auralis package
    const pythonPath = this.isDevelopment
      ? path.join(__dirname, '..')
      : process.resourcesPath;

    const fullPythonPath = pythonPath + (process.env.PYTHONPATH ? path.delimiter + process.env.PYTHONPATH : '');
    console.log(`Python path for auralis: ${pythonPath}`);
    console.log(`Full PYTHONPATH: ${fullPythonPath}`);
    console.log(`Is development mode: ${this.isDevelopment}`);

    this.pythonProcess = spawn(pythonCmd, pythonArgs, {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: cwd,
      detached: true,  // Create new process group for easy cleanup
      env: {
        // buildBackendEnv() clears AURALIS_DEV_MODE on the production path
        // so an ambient value from the parent shell can't silently reopen
        // the dev-mode CORS/WS origin allowlist in a packaged build (#4898).
        ...buildBackendEnv(process.env, this.isDevelopment),
        PYTHONUNBUFFERED: '1',
        // Tell backend it's running in Electron
        ELECTRON_MODE: '1',
        // Add auralis package to Python path
        PYTHONPATH: fullPythonPath
      }
    });

    return new Promise((resolve, reject) => {
      let startupOutput = '';
      let resolved = false;

      const markReady = (source) => {
        if (!resolved) {
          resolved = true;
          this.backendReady = true;
          console.log(`✓ Backend is ready! (detected from ${source})`);
          resolve();
        }
      };

      const checkReadiness = (output, source) => {
        // "Uvicorn running on" is printed AFTER successful socket bind — most reliable signal
        if (output.includes('Uvicorn running') ||
            output.includes('Application startup complete')) {
          markReady(source);
        }
      };

      // Watch stdout for readiness
      this.pythonProcess.stdout.on('data', (data) => {
        const output = data.toString();
        startupOutput += output;
        // #4920: route through electron-log (not console.log) so backend
        // output — including security-relevant lines like rejected
        // path-traversal attempts and WS origin rejections — lands in the
        // same rotated on-disk log file already used by the auto-updater,
        // instead of vanishing when there's no attached terminal in a
        // packaged build. electron-log's default transports still echo to
        // the console too, so dev-mode visibility is unchanged.
        log.info('[Backend]', output.trim());
        checkReadiness(output, 'stdout');
      });

      // Watch stderr too — uvicorn logs go to stderr via Python logging
      this.pythonProcess.stderr.on('data', (data) => {
        const output = data.toString();
        startupOutput += output;
        log.error('[Backend Error]', output.trim());
        checkReadiness(output, 'stderr');

        // Detect fatal bind errors
        if (output.includes('address already in use')) {
          if (!resolved) {
            resolved = true;
            reject(new Error('Port 8765 is already in use — backend cannot start'));
          }
        }
      });

      this.pythonProcess.on('exit', (code, signal) => {
        console.log(`Backend process exited with code ${code}, signal ${signal}`);
        if (code !== 0 && !this.isQuitting) {
          reject(new Error(`Backend exited with code ${code}`));
        }
      });

      this.pythonProcess.on('error', (error) => {
        console.error('Failed to start backend process:', error);
        reject(error);
      });

      // Timeout after 30 seconds
      setTimeout(() => {
        if (!this.backendReady) {
          console.error('Backend startup timeout. Output so far:', startupOutput);
          reject(new Error('Backend startup timeout'));
        }
      }, 30000);
    });
  }

  async waitForBackendHealth() {
    // Give backend a moment to start serving
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Try to ping the health endpoint
    try {
      const http = require('node:http');
      const options = {
        hostname: 'localhost',
        port: 8765,
        path: '/api/health',
        method: 'GET',
        timeout: 5000
      };

      await new Promise((resolve, reject) => {
        const req = http.request(options, (res) => {
          console.log(`Health check status: ${res.statusCode}`);
          if (res.statusCode === 200) {
            resolve();
          } else {
            reject(new Error(`Health check failed with status ${res.statusCode}`));
          }
        });

        req.on('error', (error) => {
          console.warn('Health check request failed:', error.message);
          // Don't fail completely, just warn
          resolve();
        });

        req.on('timeout', () => {
          console.warn('Health check timed out');
          req.destroy();
          resolve();
        });

        req.end();
      });

      console.log('✓ Backend health check passed');
    } catch (error) {
      console.warn('Backend health check warning:', error.message);
      // Continue anyway - the backend might be ready but not responding to HTTP yet
    }
  }

  async createWindow() {
    console.log('Creating main window...');

    this.mainWindow = new BrowserWindow({
      width: 1400,
      height: 900,
      minWidth: 800,
      minHeight: 600,
      show: false, // Don't show until ready
      titleBarStyle: 'default',
      backgroundColor: '#1a1a1a',
      icon: path.join(__dirname, 'assets', 'icon.png'),
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        sandbox: true,
        preload: path.join(__dirname, 'preload.js'),
        webSecurity: true
      }
    });

    // Load the React app
    let startUrl;

    if (this.isDevelopment) {
      // Development: Load from Vite dev server (if running) or backend
      startUrl = 'http://localhost:3000'; // Try React dev server first
    } else {
      // Production: Backend serves the built React app
      startUrl = 'http://localhost:8765';
    }

    console.log(`Loading URL: ${startUrl}`);

    // Handle navigation
    this.mainWindow.webContents.setWindowOpenHandler(({ url }) => {
      openExternalSafely(url);
      return { action: 'deny' };
    });

    try {
      await this.mainWindow.loadURL(startUrl);
      console.log('✓ URL loaded successfully');

      // Show window when ready (with timeout fallback for Linux)
      let windowShown = false;

      this.mainWindow.once('ready-to-show', () => {
        if (!windowShown) {
          windowShown = true;
          this.mainWindow.show();
          console.log('✓ Window shown (ready-to-show event)');
        }
      });

      // Fallback: Force show after 2 seconds if event doesn't fire
      // This fixes display issues on some Linux window managers
      setTimeout(() => {
        if (!windowShown) {
          windowShown = true;
          this.mainWindow.show();
          console.log('✓ Window shown (fallback timeout)');
        }
      }, 2000);

      // Open DevTools in development
      if (this.isDevelopment) {
        this.mainWindow.webContents.openDevTools();
      }

    } catch (error) {
      console.error('Failed to load URL:', error);

      // Try alternative URL if development failed
      if (this.isDevelopment) {
        console.log('Trying backend URL instead...');
        try {
          await this.mainWindow.loadURL('http://localhost:8765');
          this.mainWindow.show();
        } catch (backendError) {
          // Show error page
          this.mainWindow.loadFile(path.join(__dirname, 'error.html'));
          this.mainWindow.show();
        }
      } else {
        // Show error page in production
        this.mainWindow.loadFile(path.join(__dirname, 'error.html'));
        this.mainWindow.show();
      }
    }

    // Handle window closed
    this.mainWindow.on('closed', () => {
      this.mainWindow = null;
    });
  }

  async initialize() {
    try {
      console.log('🚀 Initializing Auralis desktop app...');
      console.log(`Mode: ${this.isDevelopment ? 'Development' : 'Production'}`);
      console.log(`Platform: ${process.platform}`);
      console.log('');

      // Start Python backend first
      console.log('[1/3] Starting backend...');
      await this.startPythonBackend();
      console.log('✓ Backend started');
      console.log('');

      // Wait for backend to be healthy
      console.log('[2/3] Checking backend health...');
      await this.waitForBackendHealth();
      console.log('✓ Backend healthy');
      console.log('');

      // Then create the UI window
      console.log('[3/3] Creating UI window...');
      await this.createWindow();
      console.log('✓ UI ready');
      console.log('');

      console.log('✅ Auralis is ready!');

    } catch (error) {
      console.error('❌ Failed to initialize app:', error);

      dialog.showErrorBox('Startup Error',
        `Failed to start Auralis:\n\n${error.message}\n\nPlease check that Python is installed and try again.`);

      this.cleanup();
      app.quit();
    }
  }

  cleanup() {
    console.log('Cleaning up...');
    this.isQuitting = true;

    if (this.pythonProcess && !this.pythonProcess.killed) {
      console.log('Terminating Python backend...');

      // Try graceful shutdown first
      this.pythonProcess.kill('SIGTERM');

      // Force kill after 2 seconds (reduced from 5)
      setTimeout(() => {
        if (this.pythonProcess && !this.pythonProcess.killed) {
          console.log('Backend still running, sending SIGKILL...');
          this.pythonProcess.kill('SIGKILL');

          // Also try to kill the process tree
          if (this.pythonProcess.pid) {
            try {
              process.kill(-this.pythonProcess.pid, 'SIGKILL');
              console.log('Killed process tree');
            } catch (e) {
              // Ignore errors (process may already be dead)
            }
          }
        }
      }, 2000);
    }

    this.pythonProcess = null;
  }
}

// Initialize app instance
const auralisApp = new AuralisApp();

// Set up IPC handlers
ipcMain.handle('select-file', async () => {
  const result = await dialog.showOpenDialog(auralisApp.mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Audio Files', extensions: ['mp3', 'wav', 'flac', 'm4a', 'ogg', 'aac'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });
  return result.filePaths;
});

ipcMain.handle('select-folder', async () => {
  const result = await dialog.showOpenDialog(auralisApp.mainWindow, {
    properties: ['openDirectory']
  });
  return result.filePaths;
});

ipcMain.handle('window-minimize', () => {
  if (auralisApp.mainWindow) {
    auralisApp.mainWindow.minimize();
  }
});

ipcMain.handle('window-maximize', () => {
  if (auralisApp.mainWindow) {
    if (auralisApp.mainWindow.isMaximized()) {
      auralisApp.mainWindow.unmaximize();
    } else {
      auralisApp.mainWindow.maximize();
    }
  }
});

ipcMain.handle('window-close', () => {
  if (auralisApp.mainWindow) {
    auralisApp.mainWindow.close();
  }
});

// Configure auto-updater
autoUpdater.autoDownload = false; // Ask user before downloading
autoUpdater.autoInstallOnAppQuit = true;

// Auto-updater event handlers
autoUpdater.on('checking-for-update', () => {
  log.info('Checking for updates...');
});

autoUpdater.on('update-available', (info) => {
  log.info('Update available:', info.version);

  // Show update notification to user
  if (auralisApp.mainWindow) {
    dialog.showMessageBox(auralisApp.mainWindow, {
      type: 'info',
      title: 'Update Available',
      message: `A new version (${info.version}) is available!`,
      detail: 'Would you like to download and install it? The app will restart after installation.',
      buttons: ['Download Update', 'Later'],
      defaultId: 0,
      cancelId: 1
    }).then((result) => {
      if (result.response === 0) {
        autoUpdater.downloadUpdate();
      }
    });
  }
});

autoUpdater.on('update-not-available', (info) => {
  log.info('Update not available. Current version:', info.version);
});

autoUpdater.on('error', (err) => {
  log.error('Auto-updater error:', err);
});

autoUpdater.on('download-progress', (progressObj) => {
  let logMessage = `Download speed: ${progressObj.bytesPerSecond} - Downloaded ${progressObj.percent}%`;
  logMessage = `${logMessage} (${progressObj.transferred}/${progressObj.total})`;
  log.info(logMessage);

  // Update UI with progress (could send to renderer via IPC)
  if (auralisApp.mainWindow) {
    auralisApp.mainWindow.webContents.send('download-progress', progressObj);
  }
});

autoUpdater.on('update-downloaded', (info) => {
  log.info('Update downloaded:', info.version);

  // Notify user that update is ready
  if (auralisApp.mainWindow) {
    dialog.showMessageBox(auralisApp.mainWindow, {
      type: 'info',
      title: 'Update Ready',
      message: 'Update downloaded successfully!',
      detail: 'The update will be installed when you quit the application. Restart now?',
      buttons: ['Restart Now', 'Later'],
      defaultId: 0,
      cancelId: 1
    }).then((result) => {
      if (result.response === 0) {
        autoUpdater.quitAndInstall();
      }
    });
  }
});

// IPC handler for manual update check
ipcMain.handle('check-for-updates', async () => {
  if (!app.isPackaged) {
    return { available: false, message: 'Updates only available in packaged app' };
  }

  try {
    const result = await autoUpdater.checkForUpdates();
    return { available: true, version: result?.updateInfo?.version };
  } catch (error) {
    log.error('Error checking for updates:', error);
    return { available: false, error: error.message };
  }
});

// App lifecycle events
app.whenReady().then(() => {
  console.log('Electron app ready');
  auralisApp.initialize();

  // Check for updates on startup (only in production)
  if (app.isPackaged) {
    setTimeout(() => {
      log.info('Checking for updates on startup...');
      autoUpdater.checkForUpdates().catch(err => {
        log.error('Failed to check for updates:', err);
      });
    }, 3000); // Wait 3 seconds after startup
  }
});

app.on('window-all-closed', () => {
  console.log('All windows closed');
  auralisApp.cleanup();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  console.log('App activated');
  if (BrowserWindow.getAllWindows().length === 0) {
    auralisApp.initialize();
  }
});

app.on('before-quit', (event) => {
  console.log('App before quit');
  if (!auralisApp.isQuitting) {
    event.preventDefault();
    auralisApp.cleanup();
    setTimeout(() => {
      app.quit();
    }, 1000);
  }
});

// Handle protocol for deep linking (future use)
app.setAsDefaultProtocolClient('auralis');

// Security: Prevent new window creation (Electron 12+ API)
app.on('web-contents-created', (_event, contents) => {
  contents.setWindowOpenHandler(({ url }) => {
    openExternalSafely(url);
    return { action: 'deny' };
  });

  // setWindowOpenHandler only governs *new* windows. In-place top-level
  // navigation of an existing window has no default restriction: with no
  // will-navigate listener Electron just goes wherever it is pointed, and
  // because preload.js is attached to the BrowserWindow rather than to a URL
  // it re-runs there, handing the whole electronAPI IPC surface to that
  // origin (#4858). Registered on web-contents-created so every webContents
  // is covered, not only mainWindow.
  //
  // will-redirect is needed as well: a 3xx to an off-origin destination does
  // not re-emit will-navigate, so checking only the latter would let a
  // redirect chain walk straight out of localhost.
  const blockOffOrigin = (event, url) => {
    // `!app.isPackaged` is how the rest of main.js decides dev mode (line 32);
    // NODE_ENV is not set in a packaged build, so reusing it here would keep
    // the :3000 dev origin allowed in production.
    if (isAllowedAppNavigation(url, { isDevelopment: !app.isPackaged })) {
      return;
    }
    event.preventDefault();
    console.warn(`Blocked in-window navigation to non-app origin: ${url}`);
    // Treat it the way a clicked external link is treated — the user's
    // browser, not this window, subject to the same scheme allowlist.
    openExternalSafely(url);
  };

  contents.on('will-navigate', blockOffOrigin);
  contents.on('will-redirect', blockOffOrigin);
});