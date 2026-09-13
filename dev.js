#!/usr/bin/env node

const { spawn } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');

class DevEnvironment {
  constructor() {
    this.processes = [];
    this.backendReady = false;
    this.frontendReady = false;
    this.isShuttingDown = false;
  }

  log(source, message) {
    const timestamp = new Date().toLocaleTimeString();
    console.log(`[${timestamp}] [${source}] ${message}`);
  }

  async checkRequirements() {
    this.log('SETUP', 'Checking requirements...');

    // Check if Python is available
    try {
      const pythonCheck = spawn('python', ['--version'], { stdio: 'pipe' });
      await new Promise((resolve, reject) => {
        pythonCheck.on('close', (code) => {
          if (code === 0) resolve();
          else reject(new Error('Python not found'));
        });
      });
      this.log('SETUP', '✓ Python is available');
    } catch (error) {
      this.log('SETUP', '✗ Python not found. Please install Python 3.8+');
      process.exit(1);
    }

    // Check if Node.js dependencies are installed
    const desktopPackageJson = path.join(__dirname, '..', 'desktop', 'package.json');
    const desktopNodeModules = path.join(__dirname, '..', 'desktop', 'node_modules');

    if (!fs.existsSync(desktopNodeModules)) {
      this.log('SETUP', 'Installing Electron dependencies...');
      await this.runCommand('pnpm', ['install'], '../desktop');
    }

    // Check backend dependencies
    const backendRequirements = path.join(__dirname, '..', 'backend', 'requirements.txt');
    if (fs.existsSync(backendRequirements)) {
      this.log('SETUP', 'Backend requirements.txt found');
      // TODO: Check if pip packages are installed
    }

    this.log('SETUP', '✓ Requirements check complete');
  }

  async startBackend() {
    return new Promise((resolve, reject) => {
      this.log('BACKEND', 'Starting Python backend...');

      const backendPath = path.join(__dirname, '..', 'auralis-web', 'backend');
      const mainPy = path.join(backendPath, 'main.py');

      if (!fs.existsSync(mainPy)) {
        this.log('BACKEND', '✗ backend/main.py not found');
        reject(new Error('Backend main.py not found'));
        return;
      }

      const backendProcess = spawn('python', [mainPy], {
        cwd: backendPath,
        stdio: ['pipe', 'pipe', 'pipe']
      });

      this.processes.push({ name: 'backend', process: backendProcess });

      let startupOutput = '';

      backendProcess.stdout.on('data', (data) => {
        const output = data.toString();
        startupOutput += output;

        // Log backend output
        output.split('\n').forEach(line => {
          if (line.trim()) {
            this.log('BACKEND', line.trim());
          }
        });

        // Look for backend ready signals
        if (output.includes('Backend ready') ||
            output.includes('Uvicorn running') ||
            output.includes('Application startup complete')) {
          this.backendReady = true;
          this.log('BACKEND', '✓ Backend is ready');
          resolve();
        }
      });

      backendProcess.stderr.on('data', (data) => {
        const error = data.toString();
        error.split('\n').forEach(line => {
          if (line.trim()) {
            this.log('BACKEND', `ERROR: ${line.trim()}`);
          }
        });
      });

      backendProcess.on('exit', (code, signal) => {
        if (code !== 0 && !this.isShuttingDown) {
          this.log('BACKEND', `✗ Backend exited with code ${code}`);
          reject(new Error(`Backend exited with code ${code}`));
        }
      });

      backendProcess.on('error', (error) => {
        this.log('BACKEND', `✗ Failed to start: ${error.message}`);
        reject(error);
      });

      // Timeout after 30 seconds
      setTimeout(() => {
        if (!this.backendReady) {
          this.log('BACKEND', '✗ Startup timeout');
          this.log('BACKEND', 'Output so far:', startupOutput);
          reject(new Error('Backend startup timeout'));
        }
      }, 30000);
    });
  }

  async startFrontend() {
    return new Promise((resolve, reject) => {
      this.log('FRONTEND', 'Starting React frontend...');

      const frontendPath = path.join(__dirname, '..', 'auralis-web', 'frontend');

      if (!fs.existsSync(frontendPath)) {
        this.log('FRONTEND', '✗ frontend directory not found');
        // For now, just resolve - we'll run without frontend in development
        this.frontendReady = true;
        resolve();
        return;
      }

      const packageJsonPath = path.join(frontendPath, 'package.json');
      if (!fs.existsSync(packageJsonPath)) {
        this.log('FRONTEND', '✗ frontend/package.json not found');
        this.frontendReady = true;
        resolve();
        return;
      }

      // Check if node_modules exists
      const nodeModulesPath = path.join(frontendPath, 'node_modules');
      if (!fs.existsSync(nodeModulesPath)) {
        this.log('FRONTEND', 'Installing frontend dependencies...');
        this.log('FRONTEND', '⚠ Frontend dependencies not installed - skipping for now');
      }

      const frontendProcess = spawn('pnpm', ['run', 'dev'], {
        cwd: frontendPath,
        stdio: ['pipe', 'pipe', 'pipe']
      });

      this.processes.push({ name: 'frontend', process: frontendProcess });

      frontendProcess.stdout.on('data', (data) => {
        const output = data.toString();
        output.split('\n').forEach(line => {
          if (line.trim()) {
            this.log('FRONTEND', line.trim());
          }
        });

        // Look for Vite ready signals
        if (output.includes('Local:') ||
            output.includes('ready in') ||
            output.includes('Local') && output.includes('3000')) {
          this.frontendReady = true;
          this.log('FRONTEND', '✓ Frontend is ready');
          resolve();
        }
      });

      frontendProcess.stderr.on('data', (data) => {
        const error = data.toString();
        error.split('\n').forEach(line => {
          if (line.trim()) {
            this.log('FRONTEND', line.trim());
          }
        });
      });

      frontendProcess.on('exit', (code) => {
        if (code !== 0 && !this.isShuttingDown) {
          this.log('FRONTEND', `✗ Frontend exited with code ${code}`);
          // Don't reject - frontend is optional in development
        }
      });

      // Resolve after 10 seconds even if we don't see ready signal
      setTimeout(() => {
        if (!this.frontendReady) {
          this.log('FRONTEND', '⚠ Assuming frontend is ready (timeout)');
          this.frontendReady = true;
          resolve();
        }
      }, 10000);
    });
  }

  async startElectron() {
    this.log('ELECTRON', 'Starting Electron...');

    const electronProcess = spawn('pnpm', ['run', 'electron'], {
      cwd: path.join(__dirname, '..', 'desktop'),
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, NODE_ENV: 'development' }
    });

    this.processes.push({ name: 'electron', process: electronProcess });

    electronProcess.stdout.on('data', (data) => {
      const output = data.toString();
      output.split('\n').forEach(line => {
        if (line.trim()) {
          this.log('ELECTRON', line.trim());
        }
      });
    });

    electronProcess.stderr.on('data', (data) => {
      const error = data.toString();
      error.split('\n').forEach(line => {
        if (line.trim()) {
          this.log('ELECTRON', line.trim());
        }
      });
    });

    electronProcess.on('exit', (code) => {
      this.log('ELECTRON', `Electron exited with code ${code}`);
      this.shutdown();
    });

    electronProcess.on('error', (error) => {
      this.log('ELECTRON', `✗ Failed to start: ${error.message}`);
      this.shutdown();
    });
  }

  async runCommand(cmd, args, cwd) {
    return new Promise((resolve, reject) => {
      const fullCwd = path.join(__dirname, cwd);
      const process = spawn(cmd, args, {
        cwd: fullCwd,
        stdio: 'inherit'
      });

      process.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`${cmd} ${args.join(' ')} failed with code ${code}`));
        }
      });

      process.on('error', (error) => {
        reject(error);
      });
    });
  }

  shutdown() {
    if (this.isShuttingDown) return;

    this.isShuttingDown = true;
    this.log('SHUTDOWN', 'Shutting down development environment...');

    this.processes.forEach(({ name, process }) => {
      if (process && !process.killed) {
        this.log('SHUTDOWN', `Terminating ${name}...`);
        process.kill('SIGTERM');
      }
    });

    // Force exit after 5 seconds
    setTimeout(() => {
      this.log('SHUTDOWN', 'Force exit');
      process.exit(0);
    }, 5000);
  }

  async start() {
    try {
      this.log('DEV', '🚀 Starting Auralis development environment...');

      // Set up signal handlers
      process.on('SIGINT', () => this.shutdown());
      process.on('SIGTERM', () => this.shutdown());

      // Check requirements
      await this.checkRequirements();

      // Start backend first
      await this.startBackend();

      // Start frontend (optional)
      await this.startFrontend();

      // Give everything a moment to settle
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Start Electron
      await this.startElectron();

      this.log('DEV', '✓ All components started successfully');

    } catch (error) {
      this.log('DEV', `✗ Failed to start: ${error.message}`);
      this.shutdown();
      process.exit(1);
    }
  }
}

// Check if running directly
if (require.main === module) {
  const devEnv = new DevEnvironment();
  devEnv.start();
}

module.exports = DevEnvironment;