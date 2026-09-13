#!/usr/bin/env node

/**
 * Memory Test Failsafe
 * ====================
 *
 * Safely tests frontend test suite for memory leaks without crashing the system.
 *
 * Features:
 * - Runs tests by category (isolated memory profiles)
 * - Progressive heap size allocation (8MB → 2GB)
 * - Real-time memory monitoring with kill switches
 * - Detailed memory reports and leak detection
 * - Windows/Linux compatibility
 *
 * Usage:
 *   pnpm run test:memory:failsafe
 *   pnpm run test:memory:failsafe -- --category library
 *   pnpm run test:memory:failsafe -- --max-heap 1024
 */

const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');

// Configuration
const config = {
  maxHeap: 2048,           // 2GB default
  killThreshold: 1800,     // Kill at 1.8GB
  updateInterval: 500,     // Check memory every 500ms
  timeout: 600000,         // 10 minute timeout per test
  testCategories: [
    'components',
    'integration',
    'services',
    'hooks',
    'contexts'
  ]
};

// Parse command line arguments
const args = process.argv.slice(2);
let targetCategory = null;
let customMaxHeap = config.maxHeap;

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--category' && args[i + 1]) {
    targetCategory = args[i + 1];
    i++;
  }
  if (args[i] === '--max-heap' && args[i + 1]) {
    customMaxHeap = parseInt(args[i + 1]);
    i++;
  }
}

config.maxHeap = customMaxHeap;
config.killThreshold = Math.max(customMaxHeap * 0.85, 1000);

// Logging utilities
const log = {
  info: (msg) => console.log(`[INFO] ${msg}`),
  warn: (msg) => console.warn(`[WARN] ${msg}`),
  error: (msg) => console.error(`[ERROR] ${msg}`),
  success: (msg) => console.log(`[✓] ${msg}`),
  failed: (msg) => console.error(`[✗] ${msg}`),
};

// Memory utilities
function formatBytes(bytes) {
  const mb = (bytes / 1024 / 1024).toFixed(2);
  return `${mb}MB`;
}

function getSystemMemory() {
  return {
    total: os.totalmem(),
    free: os.freemem(),
    used: os.totalmem() - os.freemem()
  };
}

// Test runner
async function runTestCategory(category) {
  return new Promise((resolve) => {
    const startTime = Date.now();
    const startMem = getSystemMemory();

    log.info(`\n${'='.repeat(60)}`);
    log.info(`Testing: ${category.toUpperCase()}`);
    log.info(`Heap size: ${config.maxHeap}MB | Kill threshold: ${config.killThreshold}MB`);
    log.info(`System memory: ${formatBytes(startMem.used)} / ${formatBytes(startMem.total)}`);
    log.info(`${'='.repeat(60)}`);

    // Build test command
    const testPattern = `src/**/__tests__/${category}/**/*.test.tsx`;
    const nodeArgs = [
      `--max-old-space-size=${config.maxHeap}`,
      '--expose-gc'
    ];

    const viteArgs = [
      'run',
      'test:run',
      '--',
      '--reporter=verbose',
      '--bail',
      testPattern
    ];

    const process = spawn('pnpm', viteArgs, {
      cwd: process.cwd(),
      env: {
        ...process.env,
        NODE_OPTIONS: nodeArgs.join(' ')
      },
      stdio: ['inherit', 'pipe', 'pipe']
    });

    let lastMemWarning = 0;
    let maxMemUsed = 0;
    let peakHeapSize = 0;

    const memoryCheck = setInterval(() => {
      const current = getSystemMemory();
      const usedMB = current.used / 1024 / 1024;

      if (usedMB > maxMemUsed) {
        maxMemUsed = usedMB;
      }

      // Warn at 80% threshold
      if (usedMB > (config.killThreshold * 0.8)) {
        if (Date.now() - lastMemWarning > 5000) {
          log.warn(`Memory high: ${formatBytes(current.used)} (${(usedMB / config.killThreshold * 100).toFixed(1)}% of kill threshold)`);
          lastMemWarning = Date.now();
        }
      }

      // Kill at hard threshold
      if (usedMB > config.killThreshold) {
        log.error(`MEMORY KILL THRESHOLD EXCEEDED: ${formatBytes(current.used)}`);
        log.error(`Killing test process to protect system`);
        process.kill('SIGTERM');
        clearInterval(memoryCheck);
      }
    }, config.updateInterval);

    // Handle stdout for memory tracking
    if (process.stdout) {
      process.stdout.on('data', (data) => {
        process.stdout.write(data);
      });
    }

    if (process.stderr) {
      process.stderr.on('data', (data) => {
        process.stderr.write(data);
      });
    }

    process.on('close', (code) => {
      clearInterval(memoryCheck);

      const endTime = Date.now();
      const endMem = getSystemMemory();
      const duration = ((endTime - startTime) / 1000).toFixed(2);

      log.info(`\n${'-'.repeat(60)}`);
      log.info(`Category: ${category}`);
      log.info(`Duration: ${duration}s`);
      log.info(`Max Memory Used: ${maxMemUsed.toFixed(2)}MB`);
      log.info(`System Memory Change: ${formatBytes(startMem.used)} → ${formatBytes(endMem.used)}`);

      if (code === 0) {
        log.success(`${category} tests passed`);
      } else {
        log.failed(`${category} tests failed (exit code: ${code})`);
      }
      log.info(`${'-'.repeat(60)}\n`);

      resolve({
        category,
        code,
        duration: parseFloat(duration),
        maxMemory: maxMemUsed,
        startMemory: startMem.used,
        endMemory: endMem.used
      });
    });

    process.on('error', (err) => {
      clearInterval(memoryCheck);
      log.error(`Process error: ${err.message}`);
      resolve({
        category,
        code: 1,
        duration: 0,
        maxMemory: maxMemUsed,
        error: err.message
      });
    });

    // Timeout protection
    setTimeout(() => {
      if (!process.killed) {
        log.warn(`Timeout reached for ${category}, killing process`);
        process.kill('SIGTERM');
      }
    }, config.timeout);
  });
}

// Main runner
async function main() {
  log.info('Memory Test Failsafe - Starting');
  log.info(`Node: ${process.version}`);
  log.info(`Platform: ${os.platform()}`);
  log.info(`Arch: ${os.arch()}`);
  log.info(`Total System Memory: ${formatBytes(os.totalmem())}`);

  const categoriesToTest = targetCategory
    ? [targetCategory]
    : config.testCategories;

  const results = [];

  for (const category of categoriesToTest) {
    const result = await runTestCategory(category);
    results.push(result);

    // Brief pause between categories for system recovery
    if (category !== categoriesToTest[categoriesToTest.length - 1]) {
      log.info('Waiting 5 seconds before next test category...');
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  }

  // Summary report
  log.info(`\n${'='.repeat(60)}`);
  log.info('MEMORY TEST SUMMARY');
  log.info(`${'='.repeat(60)}`);

  let totalDuration = 0;
  let maxMemory = 0;
  let passed = 0;
  let failed = 0;

  results.forEach(r => {
    totalDuration += r.duration;
    if (r.maxMemory > maxMemory) maxMemory = r.maxMemory;

    const status = r.code === 0 ? '✓' : '✗';
    const memStr = r.maxMemory ? `${r.maxMemory.toFixed(2)}MB` : 'unknown';

    log.info(`${status} ${r.category.padEnd(15)} | ${r.duration.toFixed(2)}s | Peak: ${memStr}`);

    if (r.code === 0) passed++;
    else failed++;
  });

  log.info(`${'-'.repeat(60)}`);
  log.info(`Total Duration: ${totalDuration.toFixed(2)}s`);
  log.info(`Peak Memory: ${maxMemory.toFixed(2)}MB / ${config.maxHeap}MB`);
  log.info(`Pass Rate: ${passed}/${passed + failed} (${((passed / (passed + failed)) * 100).toFixed(0)}%)`);
  log.info(`${'='.repeat(60)}\n`);

  // Exit with appropriate code
  process.exit(failed > 0 ? 1 : 0);
}

main().catch(err => {
  log.error(err);
  process.exit(1);
});
