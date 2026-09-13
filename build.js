#!/usr/bin/env node

/**
 * Auralis Build Script
 * ~~~~~~~~~~~~~~~~~~~~
 *
 * Comprehensive build system that:
 * 1. Builds React frontend (production build)
 * 2. Bundles Python backend with PyInstaller
 * 3. Prepares everything for Electron packaging
 *
 * @copyright (C) 2024 Auralis Team
 * @license GPLv3
 */

const { spawn } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');

class BuildManager {
  constructor() {
    this.rootDir = __dirname;
    this.frontendDir = path.join(this.rootDir, 'auralis-web', 'frontend');
    this.backendDir = path.join(this.rootDir, 'auralis-web', 'backend');
    this.desktopDir = path.join(this.rootDir, 'desktop');
    this.distDir = path.join(this.rootDir, 'dist');
    this.vendorDspDir = path.join(this.rootDir, 'vendor', 'auralis-dsp');

    this.frontendBuildDir = path.join(this.frontendDir, 'dist');
    this.backendDistDir = path.join(this.rootDir, 'dist', 'auralis-backend');
  }

  log(category, message) {
    const timestamp = new Date().toLocaleTimeString();
    console.log(`[${timestamp}] [${category}] ${message}`);
  }

  async runCommand(cmd, args, cwd, description, extraEnv) {
    return new Promise((resolve, reject) => {
      this.log('BUILD', `${description}...`);

      const process = spawn(cmd, args, {
        cwd: cwd || this.rootDir,
        stdio: 'inherit',
        shell: true,
        env: extraEnv ? { ...require('node:process').env, ...extraEnv } : undefined
      });

      process.on('close', (code) => {
        if (code === 0) {
          this.log('BUILD', `✓ ${description} complete`);
          resolve();
        } else {
          this.log('BUILD', `✗ ${description} failed with code ${code}`);
          reject(new Error(`${description} failed`));
        }
      });

      process.on('error', (error) => {
        this.log('BUILD', `✗ ${description} error: ${error.message}`);
        reject(error);
      });
    });
  }

  async checkRequirements() {
    this.log('BUILD', '🔍 Checking requirements...');

    // Check Node.js
    try {
      await this.runCommand('node', ['--version'], this.rootDir, 'Node.js version check');
    } catch (error) {
      throw new Error('Node.js not found. Please install Node.js 16+');
    }

    // Check Python
    try {
      await this.runCommand('python', ['--version'], this.rootDir, 'Python version check');
    } catch (error) {
      throw new Error('Python not found. Please install Python 3.13+');
    }

    // Check maturin (for Rust DSP)
    try {
      await this.runCommand('maturin', ['--version'], this.rootDir, 'maturin version check');
    } catch (error) {
      this.log('BUILD', '⚠️  maturin not found, installing...');
      await this.runCommand('pip', ['install', 'maturin'], this.rootDir, 'Installing maturin');
    }

    // Check PyInstaller
    try {
      await this.runCommand('pyinstaller', ['--version'], this.rootDir, 'PyInstaller version check');
    } catch (error) {
      this.log('BUILD', '⚠️  PyInstaller not found, installing...');
      await this.runCommand('pip', ['install', 'pyinstaller'], this.rootDir, 'Installing PyInstaller');
    }

    this.log('BUILD', '✓ All requirements met');
  }

  async cleanDist() {
    this.log('BUILD', '🧹 Cleaning previous builds...');

    const dirsToClean = [
      this.frontendBuildDir,
      this.backendDistDir,
      path.join(this.backendDir, 'build'),
      path.join(this.backendDir, '*.spec'),
      this.distDir
    ];

    for (const dir of dirsToClean) {
      if (fs.existsSync(dir)) {
        if (dir.includes('*')) {
          // Handle glob patterns for spec files
          const dirPath = path.dirname(dir);
          const pattern = path.basename(dir);
          if (fs.existsSync(dirPath)) {
            const files = fs.readdirSync(dirPath);
            files.forEach(file => {
              if (file.endsWith('.spec')) {
                fs.unlinkSync(path.join(dirPath, file));
              }
            });
          }
        } else {
          fs.rmSync(dir, { recursive: true, force: true });
        }
        this.log('BUILD', `  Removed ${path.basename(dir)}`);
      }
    }

    this.log('BUILD', '✓ Clean complete');
  }

  async buildFrontend() {
    this.log('BUILD', '⚛️  Building React frontend...');

    // Check if frontend directory exists
    if (!fs.existsSync(this.frontendDir)) {
      this.log('BUILD', '⚠️  Frontend directory not found, skipping');
      return;
    }

    // Install dependencies if needed
    if (!fs.existsSync(path.join(this.frontendDir, 'node_modules'))) {
      await this.runCommand('pnpm', ['install'], this.frontendDir, 'Installing frontend dependencies');
    }

    // Build frontend
    await this.runCommand('pnpm', ['run', 'build'], this.frontendDir, 'Building frontend');

    // Verify build
    if (!fs.existsSync(this.frontendBuildDir)) {
      throw new Error('Frontend build failed - build directory not created');
    }

    this.log('BUILD', '✓ Frontend built successfully');
  }

  async buildRustDsp() {
    this.log('BUILD', '🦀 Building Rust DSP module...');

    if (!fs.existsSync(this.vendorDspDir)) {
      this.log('BUILD', '⚠️  Rust DSP directory not found, skipping');
      return;
    }

    // Build with maturin (release mode, ABI3 compat for Python 3.13)
    await this.runCommand(
      'maturin',
      ['build', '--release'],
      this.vendorDspDir,
      'Building Rust DSP with maturin',
      { PYO3_USE_ABI3_FORWARD_COMPATIBILITY: '1' }
    );

    // Install the built wheel
    const wheelsDir = path.join(this.vendorDspDir, 'target', 'wheels');
    if (fs.existsSync(wheelsDir)) {
      const wheels = fs.readdirSync(wheelsDir).filter(f => f.endsWith('.whl'));
      if (wheels.length > 0) {
        await this.runCommand(
          'pip',
          ['install', path.join(wheelsDir, wheels[wheels.length - 1]), '--force-reinstall'],
          this.rootDir,
          'Installing Rust DSP wheel'
        );
      }
    }

    this.log('BUILD', '✓ Rust DSP built and installed');
  }

  async buildBackend() {
    this.log('BUILD', '🐍 Bundling Python backend with PyInstaller...');

    // Canonical spec lives in auralis-web/backend/ (matches CI). Its relative
    // datas paths (../../auralis etc.) resolve from the spec's own location
    // (SPECPATH), so PyInstaller must run with cwd=backendDir; --distpath/
    // --workpath redirect the output back to the root dist/ this script
    // expects everywhere else (this.backendDistDir).
    const specPath = path.join(this.backendDir, 'auralis-backend.spec');
    if (!fs.existsSync(specPath)) {
      throw new Error('auralis-backend.spec not found in auralis-web/backend/');
    }

    await this.runCommand(
      'pyinstaller',
      [
        '--clean',
        '--noconfirm',
        '--distpath', this.distDir,
        '--workpath', path.join(this.backendDir, 'build'),
        'auralis-backend.spec'
      ],
      this.backendDir,
      'Bundling backend with PyInstaller'
    );

    // Verify build
    if (!fs.existsSync(this.backendDistDir)) {
      throw new Error('Backend build failed - dist directory not created');
    }

    this.log('BUILD', '✓ Backend bundled successfully');
  }

  async prepareElectronResources() {
    this.log('BUILD', '📦 Preparing Electron resources...');

    // Ensure directories exist
    const resourcesDir = path.join(this.desktopDir, 'resources');
    fs.mkdirSync(resourcesDir, { recursive: true });

    // Copy frontend build if it exists
    if (fs.existsSync(this.frontendBuildDir)) {
      const frontendResourceDir = path.join(resourcesDir, 'frontend');
      if (fs.existsSync(frontendResourceDir)) {
        fs.rmSync(frontendResourceDir, { recursive: true, force: true });
      }
      this.copyRecursive(this.frontendBuildDir, frontendResourceDir);
      this.log('BUILD', '  ✓ Frontend resources copied');
    }

    // Copy backend build (COLLECT directory with binary + _internal)
    if (fs.existsSync(this.backendDistDir)) {
      const backendResourceDir = path.join(resourcesDir, 'backend');
      if (fs.existsSync(backendResourceDir)) {
        fs.rmSync(backendResourceDir, { recursive: true, force: true });
      }
      this.copyRecursive(this.backendDistDir, backendResourceDir);
      this.log('BUILD', '  ✓ Backend resources copied');
    }

    // Copy auralis core package
    const auralisDir = path.join(this.rootDir, 'auralis');
    if (fs.existsSync(auralisDir)) {
      const auralisResourceDir = path.join(resourcesDir, 'auralis');
      if (fs.existsSync(auralisResourceDir)) {
        fs.rmSync(auralisResourceDir, { recursive: true, force: true });
      }
      this.copyRecursive(auralisDir, auralisResourceDir);
      this.log('BUILD', '  ✓ Auralis core resources copied');
    }

    this.log('BUILD', '✓ Electron resources prepared');
  }

  copyRecursive(src, dest) {
    const stats = fs.statSync(src);

    if (stats.isDirectory()) {
      fs.mkdirSync(dest, { recursive: true });
      const files = fs.readdirSync(src);

      files.forEach(file => {
        const srcPath = path.join(src, file);
        const destPath = path.join(dest, file);
        this.copyRecursive(srcPath, destPath);
      });
    } else {
      fs.copyFileSync(src, dest);
    }
  }

  async build() {
    try {
      this.log('BUILD', '🚀 Starting Auralis build process...');
      this.log('BUILD', '');

      // Step 1: Check requirements
      await this.checkRequirements();
      console.log('');

      // Step 2: Clean previous builds
      await this.cleanDist();
      console.log('');

      // Step 3: Build Rust DSP
      await this.buildRustDsp();
      console.log('');

      // Step 4: Build frontend
      await this.buildFrontend();
      console.log('');

      // Step 5: Build backend
      await this.buildBackend();
      console.log('');

      // Step 6: Prepare Electron resources
      await this.prepareElectronResources();
      console.log('');

      this.log('BUILD', '✅ Build complete!');
      this.log('BUILD', '');
      this.log('BUILD', 'Next steps:');
      this.log('BUILD', '  cd desktop && pnpm run build:linux    # AppImage');
      this.log('BUILD', '  cd desktop && pnpm run build:mac      # DMG');
      this.log('BUILD', '  cd desktop && pnpm run build:win      # NSIS installer');
      console.log('');

    } catch (error) {
      this.log('BUILD', `❌ Build failed: ${error.message}`);
      process.exit(1);
    }
  }
}

// Run build if executed directly
if (require.main === module) {
  const buildManager = new BuildManager();
  buildManager.build();
}

module.exports = BuildManager;